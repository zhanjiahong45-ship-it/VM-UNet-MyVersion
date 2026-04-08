import cv2
import numpy as np
import os
import glob


def morphological_core_segmentation(input_dir, output_dir):
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob.glob(os.path.join(input_dir, '*.[jp][pn]*[g]'))

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)

        img = cv2.imread(img_path)
        if img is None: continue

        h, w = img.shape[:2]
        img_center = (w / 2, h / 2)
        max_dist = np.sqrt(img_center[0] ** 2 + img_center[1] ** 2)

        # ==========================================
        # 1. FOV 处理 (基础：只过滤四周死黑的角)
        # ==========================================
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, fov_mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        kernel_fov = np.ones((11, 11), np.uint8)
        fov_mask = cv2.morphologyEx(fov_mask, cv2.MORPH_CLOSE, kernel_fov)

        # ==========================================
        # 2. 图像增强与反转 (拉升对比度)
        # ==========================================
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        inv_gray = cv2.bitwise_not(enhanced_gray)
        inv_gray = cv2.bitwise_and(inv_gray, inv_gray, mask=fov_mask)

        # ==========================================
        # 3. 阈值分割 (获取初始 Mask)
        # ==========================================
        valid_pixels = inv_gray[fov_mask == 255]
        if len(valid_pixels) == 0: continue
        ret, _ = cv2.threshold(valid_pixels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        _, binary_mask = cv2.threshold(inv_gray, ret, 255, cv2.THRESH_BINARY)

        # ==========================================
        # 4. 核心：基于“生物几何形态先验”的智能筛选
        # ==========================================
        # 寻找所有连通块的轮廓 (不关心位置)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = -1

        # 找到图像中所有有效色块的最大面积，用于归一化
        max_possible_area = h * w

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # 过滤极小噪点 (小于全图面积的 0.1%)
            if area < (max_possible_area * 0.001):
                continue

            # --- 形态学指标计算 ---

            # [指标 A] 长宽比 (Aspect Ratio) - 越接近 1 越圆/团块状
            x, y, w_b, h_b = cv2.boundingRect(cnt)
            major_axis = max(w_b, h_b)
            minor_axis = min(w_b, h_b) + 1e-5  # 防止除零
            aspect_ratio = major_axis / minor_axis
            roundness_score = 1.0 / aspect_ratio  # 越接近 1 越饱满

            # [指标 B] 实心度 (Solidity) - 越接近 1 越充实
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-5)  # 越接近 1 越饱满

            # --- 智能打分系统 ---
            # 计算质心
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # 距离中心越近越好，作为一个很弱的参考项
                dist_norm = np.sqrt((cx - img_center[0]) ** 2 + (cy - img_center[1]) ** 2) / max_dist
            else:
                dist_norm = 1.0

            # 综合得分函数：追求 (面积大 + 形状饱满 + 实心度高)。
            # 位置得分 (1.0 - dist_norm * 0.2) 依然存在，但权重被我压得很低，
            # 即使长在边缘，只要又大又圆，照样能拿高分！
            score = (area / max_possible_area) * roundness_score * solidity * (1.0 - dist_norm * 0.2)

            if score > best_score:
                best_score = score
                best_contour = cnt

        # ==========================================
        # 5. 生成最终并平滑掩膜
        # ==========================================
        final_mask = np.zeros_like(gray)
        if best_contour is not None:
            cv2.drawContours(final_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

        # 形态学平滑：用来缝合毛发割裂的边缘，熨平轮廓
        # kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        # final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_smooth)

        cv2.imwrite(out_path, final_mask)
        print(f"已重构生成: {filename}")


if __name__ == "__main__":
    INPUT_PATH = r"D:\HUBU.zhan\GDM\VM-UNet-main\inputs"
    OUTPUT_PATH = r"D:\HUBU.zhan\GDM\VM-UNet-main\outputs"
    morphological_core_segmentation(INPUT_PATH, OUTPUT_PATH)