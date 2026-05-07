"""
浅色病灶筛选工具（增强版）
==========================
结合 GT mask 形态学 + 原图颜色特征，筛选：
  1. 浅色/低对比度病灶（与周围皮肤色差小）
  2. 多点开花型（mask 中多个分散连通区域）
  3. 甜甜圈型（mask 内部有空洞）

适配 ISIC 数据集命名：
  images/  ISIC_XXXXXXX.jpg
  masks/   ISIC_XXXXXXX_segmentation.png

用法：
  python filter_lesions_v2.py

  或自定义路径：
  python filter_lesions_v2.py --image_dir "D:/your/images" --mask_dir "D:/your/masks" --output_dir "./output"

依赖：
  pip install opencv-python numpy scikit-image pandas matplotlib
"""

import os
import cv2
import numpy as np
import argparse
import csv
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# ★★★ 在这里修改你的路径 ★★★
# ============================================================
DEFAULT_IMAGE_DIR = r"D:\HUBU.zhan\GDM\VM-UNet-main\TestData\images"
DEFAULT_MASK_DIR  = r"D:\HUBU.zhan\GDM\VM-UNet-main\TestData\masks"
DEFAULT_OUTPUT_DIR = r"D:\HUBU.zhan\GDM\VM-UNet-main\TestData\filtered_results"


# ============================================================
# 核心：单张分析
# ============================================================

def analyze_single(image_path: str, mask_path: str,
                   min_component_area_ratio: float = 0.003,
                   donut_hole_ratio_min: float = 0.008,
                   donut_hole_ratio_max: float = 0.5) -> dict:
    """
    综合分析一对 image + mask。

    返回形态学特征 + 颜色特征。
    """
    filename = os.path.basename(image_path)
    result = {"filename": filename, "mask_file": os.path.basename(mask_path)}

    # ---------- 读取 ----------
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)

    if mask is None:
        result["error"] = "mask 无法读取"
        return result
    if image is None:
        result["error"] = "image 无法读取"
        return result

    # 确保尺寸一致
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 二值化
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    total_pixels = binary.shape[0] * binary.shape[1]
    foreground_pixels = int(np.sum(binary > 0))

    if foreground_pixels < 100:
        result.update({"pattern_type": "empty", "foreground_ratio": 0,
                       "is_multifocal": False, "is_donut": False, "is_pale": False})
        return result

    result["foreground_ratio"] = round(foreground_pixels / total_pixels, 4)

    # ============================================================
    # A. 形态学分析
    # ============================================================

    # --- A1. 连通区域（多点开花）---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = int(total_pixels * min_component_area_ratio)

    valid_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_components.append({
                "label": i, "area": int(area),
                "cx": float(centroids[i][0]), "cy": float(centroids[i][1]),
            })

    num_valid = len(valid_components)
    result["num_components"] = num_valid

    # 多点开花判定：>=2 个区域，第二大至少是最大的 3%
    is_multifocal = False
    if num_valid >= 2:
        areas_sorted = sorted([c["area"] for c in valid_components], reverse=True)
        ratio_2nd = areas_sorted[1] / areas_sorted[0]
        result["second_largest_ratio"] = round(ratio_2nd, 4)
        if ratio_2nd >= 0.03:
            is_multifocal = True
    else:
        result["second_largest_ratio"] = 0

    result["is_multifocal"] = is_multifocal

    # 分散度（质心间平均距离 / 图像对角线）
    diag = np.sqrt(binary.shape[0]**2 + binary.shape[1]**2)
    if num_valid >= 2:
        dists = []
        for i in range(len(valid_components)):
            for j in range(i+1, len(valid_components)):
                d = np.sqrt((valid_components[i]["cx"] - valid_components[j]["cx"])**2 +
                            (valid_components[i]["cy"] - valid_components[j]["cy"])**2)
                dists.append(d)
        result["dispersion"] = round(np.mean(dists) / diag, 4)  # 归一化
    else:
        result["dispersion"] = 0

    # --- A2. 空洞检测（甜甜圈）---
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for idx, h in enumerate(hierarchy):
            parent = h[3]
            if parent != -1:
                hole_area = cv2.contourArea(contours[idx])
                if (hole_area >= foreground_pixels * donut_hole_ratio_min and
                    hole_area <= foreground_pixels * donut_hole_ratio_max):
                    holes.append(int(hole_area))

    is_donut = len(holes) > 0
    result["is_donut"] = is_donut
    result["num_holes"] = len(holes)
    result["hole_area_ratio"] = round(sum(holes) / foreground_pixels, 4) if holes else 0

    # --- A3. 紧凑度 / 不规则度 ---
    all_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if all_contours:
        largest = max(all_contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(largest, True)
        area = cv2.contourArea(largest)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            result["circularity"] = round(circularity, 4)
        else:
            result["circularity"] = 0
    else:
        result["circularity"] = 0

    # ============================================================
    # B. 颜色分析（★ 关键：找浅色/低对比度病灶）
    # ============================================================

    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    fg_mask = binary > 0
    bg_mask = binary == 0

    # 排除图像边缘的黑色区域（圆形皮肤镜图像常见）
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, non_black = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    bg_mask = bg_mask & (non_black > 0)

    if np.sum(fg_mask) < 50 or np.sum(bg_mask) < 50:
        result.update({"contrast_L": 0, "contrast_A": 0, "contrast_B": 0,
                       "is_pale": False, "color_distance": 0, "fg_saturation": 0})
    else:
        # Lab 各通道前景 vs 背景均值
        fg_L = np.mean(image_lab[:,:,0][fg_mask])
        fg_A = np.mean(image_lab[:,:,1][fg_mask])
        fg_B = np.mean(image_lab[:,:,2][fg_mask])
        bg_L = np.mean(image_lab[:,:,0][bg_mask])
        bg_A = np.mean(image_lab[:,:,1][bg_mask])
        bg_B = np.mean(image_lab[:,:,2][bg_mask])

        # Lab 色差 (Delta E 简化版)
        delta_L = fg_L - bg_L
        delta_A = fg_A - bg_A
        delta_B = fg_B - bg_B
        color_distance = np.sqrt(delta_L**2 + delta_A**2 + delta_B**2)

        result["contrast_L"] = round(float(delta_L), 2)       # 亮度差（负=病灶更暗，正=更亮）
        result["contrast_A"] = round(float(delta_A), 2)       # 红绿差
        result["contrast_B"] = round(float(delta_B), 2)       # 黄蓝差
        result["color_distance"] = round(float(color_distance), 2)  # ★ 总色差

        # 前景饱和度
        fg_sat = np.mean(image_hsv[:,:,1][fg_mask])
        bg_sat = np.mean(image_hsv[:,:,1][bg_mask])
        result["fg_saturation"] = round(float(fg_sat), 2)
        result["bg_saturation"] = round(float(bg_sat), 2)
        result["sat_diff"] = round(float(fg_sat - bg_sat), 2)

        # 前景亮度
        result["fg_lightness"] = round(float(fg_L), 2)
        result["bg_lightness"] = round(float(bg_L), 2)

        # ★ 浅色病灶判定：
        #   - 色差小（与周围皮肤接近）
        #   - 或者病灶区域亮度高（偏白/粉）
        #   - 饱和度差异小
        is_pale = (color_distance < 18) or (fg_L > 160 and color_distance < 30)
        result["is_pale"] = bool(is_pale)

    # ============================================================
    # C. 综合分类
    # ============================================================
    tags = []
    if result.get("is_pale", False):
        tags.append("pale")
    if is_multifocal:
        tags.append("multifocal")
    if is_donut:
        tags.append("donut")

    result["pattern_type"] = "+".join(tags) if tags else "normal"
    result["target_score"] = compute_similarity_score(result)

    return result


def compute_similarity_score(r: dict) -> float:
    """
    计算与目标模式（浅色+多点开花/甜甜圈）的相似度得分。
    分数越高越像你要找的那种。
    """
    score = 0.0

    # 1. 浅色/低对比度（最重要，权重最高）
    cd = r.get("color_distance", 999)
    if cd < 10:
        score += 40
    elif cd < 18:
        score += 30
    elif cd < 25:
        score += 20
    elif cd < 35:
        score += 10

    # 亮度偏高额外加分
    fg_l = r.get("fg_lightness", 0)
    if fg_l > 170:
        score += 10
    elif fg_l > 150:
        score += 5

    # 2. 多点开花
    if r.get("is_multifocal", False):
        score += 25
        # 分散度越高越像
        disp = r.get("dispersion", 0)
        if disp > 0.2:
            score += 10
        elif disp > 0.1:
            score += 5

    # 3. 甜甜圈
    if r.get("is_donut", False):
        score += 20
        hr = r.get("hole_area_ratio", 0)
        if 0.03 < hr < 0.3:
            score += 10  # 空洞大小适中

    # 4. 轻微的不规则形状加分
    circ = r.get("circularity", 1)
    if circ < 0.5:
        score += 5

    return round(score, 1)


# ============================================================
# 文件名匹配
# ============================================================

def find_mask_for_image(image_path: str, mask_dir: str) -> str:
    """
    ISIC 命名规则：
      image:  ISIC_0012169.jpg
      mask:   ISIC_0012169_segmentation.png
    """
    stem = Path(image_path).stem  # ISIC_0012169
    mask_dir = Path(mask_dir)

    # 尝试多种可能的命名
    candidates = [
        mask_dir / f"{stem}_segmentation.png",
        mask_dir / f"{stem}_Segmentation.png",
        mask_dir / f"{stem}_seg.png",
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}_mask.png",
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    # 模糊匹配：找包含 stem 的文件
    for f in mask_dir.glob(f"{stem}*"):
        if f.suffix.lower() in (".png", ".jpg", ".bmp", ".tif"):
            return str(f)

    return None


# ============================================================
# 批量处理
# ============================================================

def process_one(args):
    """多进程 wrapper"""
    img_path, mask_path, kwargs = args
    try:
        return analyze_single(img_path, mask_path, **kwargs)
    except Exception as e:
        return {"filename": os.path.basename(img_path), "error": str(e)}


def batch_process(image_dir: str, mask_dir: str, output_dir: str,
                  num_workers: int = 8, **kwargs):
    """批量扫描 + 排序 + 输出"""

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集 image-mask 对
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in extensions
    ])

    print(f"[INFO] 找到 {len(image_files)} 张原图")

    pairs = []
    missing = 0
    for img_f in image_files:
        mask_f = find_mask_for_image(str(img_f), str(mask_dir))
        if mask_f:
            pairs.append((str(img_f), mask_f))
        else:
            missing += 1

    print(f"[INFO] 成功匹配 {len(pairs)} 对 image-mask（{missing} 张缺少 mask）")

    if not pairs:
        print("[ERROR] 没有匹配的 image-mask 对，请检查路径和文件命名")
        return

    # 并行分析
    tasks = [(img, msk, kwargs) for img, msk in pairs]
    results = []

    print(f"[INFO] 使用 {num_workers} 进程并行分析...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 200 == 0 or done == len(tasks):
                print(f"  进度: {done}/{len(tasks)}")
            results.append(future.result())

    # 过滤错误
    errors = [r for r in results if "error" in r]
    valid = [r for r in results if "error" not in r]

    if errors:
        print(f"[WARN] {len(errors)} 张分析出错")

    # ★ 按相似度得分排序
    valid.sort(key=lambda x: x.get("target_score", 0), reverse=True)

    # 统计
    pale = [r for r in valid if r.get("is_pale")]
    multifocal = [r for r in valid if r.get("is_multifocal")]
    donut = [r for r in valid if r.get("is_donut")]
    pale_multi = [r for r in valid if r.get("is_pale") and r.get("is_multifocal")]
    pale_donut = [r for r in valid if r.get("is_pale") and r.get("is_donut")]

    print(f"\n{'='*60}")
    print(f"  分析完成！总计: {len(valid)} 张")
    print(f"  浅色/低对比度病灶:              {len(pale)} 张")
    print(f"  多点开花 (multifocal):          {len(multifocal)} 张")
    print(f"  甜甜圈 (donut):                 {len(donut)} 张")
    print(f"  ★ 浅色 + 多点开花:              {len(pale_multi)} 张")
    print(f"  ★ 浅色 + 甜甜圈:                {len(pale_donut)} 张")
    print(f"{'='*60}")
    print(f"\n  最相似样本（按 target_score 排序）:")
    for i, r in enumerate(valid[:min(10, len(valid))]):
        print(f"    {i+1:3d}. {r['filename']:30s}  score={r['target_score']:5.1f}  "
              f"色差={str(r.get('color_distance','?')):>6s}  "
              f"{'浅色' if r.get('is_pale') else '    '}  "
              f"{'多点' if r.get('is_multifocal') else '    '}  "
              f"{'甜圈' if r.get('is_donut') else '    '}")
    if len(valid) > 10:
        print(f"    ... 完整列表见 CSV 报告")

    # ============================================================
    # 输出文件
    # ============================================================

    # 1. 完整 CSV 报告
    csv_path = output_dir / "analysis_report.csv"
    fieldnames = [
        "filename", "mask_file", "pattern_type", "target_score",
        "color_distance", "contrast_L", "fg_lightness", "bg_lightness",
        "fg_saturation", "sat_diff",
        "num_components", "second_largest_ratio", "dispersion",
        "num_holes", "hole_area_ratio", "circularity",
        "foreground_ratio", "is_pale", "is_multifocal", "is_donut",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in valid:
            writer.writerow(r)
    print(f"\n[OUTPUT] 完整报告: {csv_path}")

    # 2. 只复制命中至少一个标签的（浅色/多点开花/甜甜圈）
    matched = [r for r in valid if r.get("is_pale") or r.get("is_multifocal") or r.get("is_donut")]
    matched_img_dir = output_dir / "images"
    matched_mask_dir = output_dir / "masks"
    matched_img_dir.mkdir(parents=True, exist_ok=True)
    matched_mask_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for r in matched:
        src_img = image_dir / r["filename"]
        if src_img.exists():
            shutil.copy2(src_img, matched_img_dir / r["filename"])
        src_mask = mask_dir / r["mask_file"]
        if src_mask.exists():
            shutil.copy2(src_mask, matched_mask_dir / r["mask_file"])
        copied += 1

    print(f"[OUTPUT] 已复制 {copied} 张匹配图片到: {output_dir}")

    # 3. 生成可视化预览（可选）
    try:
        generate_preview(valid[:min(20, len(valid))], image_dir, mask_dir, output_dir)
    except Exception as e:
        print(f"[WARN] 预览生成失败: {e}")

    return valid


def generate_preview(top_results, image_dir, mask_dir, output_dir):
    """生成 Top-N 的拼图预览"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(top_results)
    if n == 0:
        return

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
    if rows == 1:
        axes = [axes]

    for idx, r in enumerate(top_results):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]

        img_path = Path(image_dir) / r["filename"]
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 叠加 mask 轮廓
            mask_path = Path(mask_dir) / r["mask_file"]
            if mask_path.exists():
                msk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if msk.shape[:2] != img.shape[:2]:
                    msk = cv2.resize(msk, (img.shape[1], img.shape[0]))
                _, msk_bin = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(msk_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, contours, -1, (255, 0, 0), 2)

            ax.imshow(img)

        tags = []
        if r.get("is_pale"): tags.append("浅")
        if r.get("is_multifocal"): tags.append("多点")
        if r.get("is_donut"): tags.append("圈")
        tag_str = "+".join(tags) if tags else "-"

        ax.set_title(f"#{idx+1} s={r['target_score']}\n{tag_str} cd={r.get('color_distance','?')}",
                     fontsize=8)
        ax.axis("off")

    # 隐藏多余的子图
    for idx in range(n, rows * cols):
        row, col = idx // cols, idx % cols
        ax = axes[row][col] if rows > 1 else axes[col]
        ax.axis("off")

    plt.suptitle("Top similar lesions (score desc)", fontsize=12)
    plt.tight_layout()
    preview_path = Path(output_dir) / "preview_top20.png"
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OUTPUT] 预览拼图: {preview_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="浅色病灶筛选工具（形态学+颜色）")
    parser.add_argument("--image_dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--mask_dir", default=DEFAULT_MASK_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_workers", type=int, default=8)

    # 灵敏度参数
    parser.add_argument("--min_area_ratio", type=float, default=0.003)
    parser.add_argument("--min_hole_ratio", type=float, default=0.008)
    parser.add_argument("--pale_threshold", type=float, default=18,
                        help="色差阈值（低于此值判定为浅色，默认18，增大=更宽松）")

    args = parser.parse_args()

    print(f"图像目录: {args.image_dir}")
    print(f"Mask目录: {args.mask_dir}")
    print(f"输出目录: {args.output_dir}")
    print()

    batch_process(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        min_component_area_ratio=args.min_area_ratio,
        donut_hole_ratio_min=args.min_hole_ratio,
    )


if __name__ == "__main__":
    main()