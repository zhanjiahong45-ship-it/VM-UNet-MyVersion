import cv2
import numpy as np
import os
import glob


def apply_dog_edge_enhancement(img_bgr, edge_amplification=2.0):
    """
    使用高斯差分 (DoG) 与非锐化掩蔽，精准放大 CIELAB L通道中的微弱高频边缘
    不会破坏原图纹理，专治“甜甜圈陷阱”的外围浅色晕环边界
    """
    # 1. 转换色彩空间并分离 L 通道
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 2. 基础局部对比度微调 (CLAHE)
    # 使用较小的 clipLimit 避免放大背景噪点，仅做基础亮度均衡
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    # 3. 高斯差分 (DoG) 提取微弱高频信号
    # blur_small 保留了大部分细节，blur_large 仅保留宏观光照背景
    blur_small = cv2.GaussianBlur(l_clahe, (3, 3), 0)
    blur_large = cv2.GaussianBlur(l_clahe, (21, 21), 0)

    # 两者相减，剥离出纯粹的边缘和纹理细节 (转为 int16 防止负数截断)
    dog_edges = cv2.subtract(blur_small.astype(np.int16), blur_large.astype(np.int16))

    # 4. 边缘信号放大与强制注入 (Unsharp Masking)
    # 公式: L_final = L_original + alpha * Edges
    # edge_amplification 越大，最外围的浅色边界线就越深、越刺眼
    l_enhanced = l_clahe.astype(np.int16) + (edge_amplification * dog_edges)

    # 截断回合法像素范围 [0, 255] 并转回 uint8
    l_enhanced = np.clip(l_enhanced, 0, 255).astype(np.uint8)

    # 5. 合并并转回 BGR
    lab_merged = cv2.merge((l_enhanced, a, b))
    img_out = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

    return img_out


def batch_process_images(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not image_paths:
        print(f"输入目录为空，请检查：{input_dir}")
        return

    print(f"找到 {len(image_paths)} 张图像，正在执行 DoG 边缘高频注入...")

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None: continue

        # 这里的 edge_amplification=2.0 可以根据效果微调，调大边缘越黑
        enhanced_img = apply_dog_edge_enhancement(img, edge_amplification=2.0)

        filename = os.path.basename(path)
        output_path = os.path.join(output_dir, f"DoG_{filename}")
        cv2.imwrite(output_path, enhanced_img)

        if (i + 1) % 10 == 0 or (i + 1) == len(image_paths):
            print(f"已处理 {i + 1}/{len(image_paths)}: {filename}")

    print(f"处理完成！图像已保存至：{output_dir}")


if __name__ == "__main__":
    # 输入输出路径
    INPUT_DIRECTORY = r"D:\HUBU.zhan\GDM\VM-UNet-main\inputs"
    OUTPUT_DIRECTORY = r"D:\HUBU.zhan\GDM\VM-UNet-main\outputEE"

    batch_process_images(INPUT_DIRECTORY, OUTPUT_DIRECTORY)