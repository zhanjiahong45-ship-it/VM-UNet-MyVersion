"""
test_partial.py — Partial Faint 三种新变体可视化测试

从训练集 train/images/ 随机抽 10 张图，每张生成所有 9 种变体（6 现有 fade + 3 新 partial）+ 原图。
输出到 ./partial_test/ 目录，每张原图对应一张 2×5 网格对比图。

用法：
    python test_partial.py
"""

import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from utils import myFaintLesionAugmentor

# ============== 配置 ==============
TRAIN_IMG_DIR = '/root/root/VM-UNet/data/isic18/train/images'
TRAIN_MSK_DIR = '/root/root/VM-UNet/data/isic18/train/masks'
OUT_DIR = './partial_test'
N_SAMPLES = 10
SEED = 42

# 9 种变体（按生成顺序展示）
ALL_MODES = [
    'diffuse', 'donut', 'nested', 'multi', 'shattered', 'ghost',
    'partial_half', 'partial_blob', 'partial_multiblob',
]

# ============== 主流程 ==============
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    if not os.path.isdir(TRAIN_IMG_DIR):
        print(f"[ERR] train image dir not found: {TRAIN_IMG_DIR}")
        return
    if not os.path.isdir(TRAIN_MSK_DIR):
        print(f"[ERR] train mask dir not found: {TRAIN_MSK_DIR}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # 列出所有图片并随机抽 N 张
    img_files = sorted([f for f in os.listdir(TRAIN_IMG_DIR)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(img_files) < N_SAMPLES:
        print(f"[WARN] only {len(img_files)} images, using all")
        sampled = img_files
    else:
        sampled = random.sample(img_files, N_SAMPLES)

    aug = myFaintLesionAugmentor(p=1.0)

    for sample_idx, img_name in enumerate(sampled):
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)

        # 找对应 mask（ISIC18 通常 mask 后缀是 _segmentation.png 或同名）
        # 这里尝试几种常见命名
        base = os.path.splitext(img_name)[0]
        mask_candidates = [
            os.path.join(TRAIN_MSK_DIR, img_name),
            os.path.join(TRAIN_MSK_DIR, base + '.png'),
            os.path.join(TRAIN_MSK_DIR, base + '_segmentation.png'),
            os.path.join(TRAIN_MSK_DIR, base + '_mask.png'),
        ]
        mask_path = None
        for mc in mask_candidates:
            if os.path.exists(mc):
                mask_path = mc
                break

        if mask_path is None:
            print(f"[WARN] no mask for {img_name}, skipping")
            continue

        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            mask = np.expand_dims(np.array(Image.open(mask_path).convert('L')), axis=2) / 255.0
        except Exception as e:
            print(f"[WARN] failed to load {img_name}: {e}")
            continue

        # 生成所有 9 种变体
        variants = {}
        for mode in ALL_MODES:
            try:
                img_v, _ = aug.apply_specific_mode((img, mask), mode)
                variants[mode] = img_v.astype(np.uint8) if img_v.dtype != np.uint8 else img_v
            except Exception as e:
                print(f"[WARN] mode '{mode}' failed for {img_name}: {e}")
                variants[mode] = img  # 失败就放原图占位

        # 画 2×5 网格：[原图, 6 现有 fade] 上排 + [3 partial + 占位] 下排
        # 实际布局：第 1 行 5 个，第 2 行 5 个 (原图 + 9 variants = 10 cells)
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        cells = [
            ('original', img),
            ('diffuse', variants['diffuse']),
            ('donut', variants['donut']),
            ('nested', variants['nested']),
            ('multi', variants['multi']),
            ('shattered', variants['shattered']),
            ('ghost', variants['ghost']),
            ('partial_half', variants['partial_half']),
            ('partial_blob', variants['partial_blob']),
            ('partial_multiblob', variants['partial_multiblob']),
        ]

        for ax, (title, im) in zip(axes.flat, cells):
            ax.imshow(im)
            # 给 3 种新 partial 加红色标题强调
            color = 'red' if title.startswith('partial_') else 'black'
            ax.set_title(title, fontsize=11, color=color, fontweight='bold')
            ax.axis('off')

        fig.suptitle(f"Sample {sample_idx+1}: {img_name}", fontsize=13)
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"sample_{sample_idx+1:02d}_{base}.png")
        plt.savefig(out_path, bbox_inches='tight', dpi=90)
        plt.close()
        print(f"[OK] {out_path}")

    print(f"\nDone. Saved {N_SAMPLES} comparison grids to {OUT_DIR}/")


if __name__ == '__main__':
    main()