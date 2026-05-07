"""
简单病变模式可视化 - 使用OpenCV
================================
避免matplotlib版本问题，直接用OpenCV生成可视化板
"""

import os
import cv2
import numpy as np
from faint_lesion_augmentor import myFaintLesionAugmentor


def create_pattern_visualization_board(image_path, mask_path, output_path='pattern_board.png'):
    """
    创建病变模式可视化板

    Parameters:
    -----------
    image_path : str
        原始图像路径
    mask_path : str
        病变mask路径
    output_path : str
        输出图像路径
    """

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read image: {image_path}")
        return

    # 读取mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"ERROR: Cannot read mask: {mask_path}")
        # 创建示例mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w//2, h//2), (w//4, h//4), 0, 0, 360, 255, -1)
        print("INFO: Using example mask (center ellipse)")

    # 确保mask是二值的
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 创建不同模式的衰减器
    patterns = {
        'Original': None,
        'Diffuse': myFaintLesionAugmentor(p=1.0, mode_probs={'diffuse': 1.0}),
        'Boundary': myFaintLesionAugmentor(p=1.0, mode_probs={'donut': 1.0}),
        'Multifocal': myFaintLesionAugmentor(p=1.0, mode_probs={'multi': 1.0}),
        'Partial': myFaintLesionAugmentor(p=1.0, mode_probs={'partial': 1.0}),
    }

    # 设置每个面板的尺寸
    panel_size = 400
    padding = 20
    text_height = 40

    # 计算整个画布的尺寸 (2行3列)
    canvas_width = 3 * panel_size + 4 * padding
    canvas_height = 2 * panel_size + 3 * padding + text_height

    # 创建白色背景画布
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # 生成不同模式变体
    for idx, (pattern_name, augmentor) in enumerate(patterns.items()):
        row = idx // 3
        col = idx % 3

        # 计算面板位置
        x_start = col * panel_size + (col + 1) * padding
        y_start = row * panel_size + (row + 1) * padding + text_height

        if augmentor is None:
            # 原始图像
            result_img = img.copy()
            result_mask = mask_binary.copy()
        else:
            # 应用衰减
            result_img, result_mask = augmentor((img.copy(), mask_binary.copy()))

        # 调整图像大小以适应面板
        img_resized = cv2.resize(result_img, (panel_size, panel_size))

        # 叠加mask轮廓
        if result_mask.ndim == 3:
            mask_vis = (result_mask[:, :, 0] > 127).astype(np.uint8)
        else:
            mask_vis = (result_mask > 127).astype(np.uint8)

        # 在调整大小前找到轮廓
        contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 调整contour坐标以适应缩放后的图像
        scale_x = panel_size / result_mask.shape[1]
        scale_y = panel_size / result_mask.shape[0]

        # 绘制图像
        canvas[y_start:y_start+panel_size, x_start:x_start+panel_size] = img_resized

        # 绘制轮廓
        for contour in contours:
            contour_scaled = contour.copy()
            contour_scaled[:, 0, 0] = (contour[:, 0, 0] * scale_x + x_start).astype(int)
            contour_scaled[:, 0, 1] = (contour[:, 0, 1] * scale_y + y_start).astype(int)
            cv2.drawContours(canvas, [contour_scaled], -1, (0, 0, 255), 2)

        # 添加标题
        title_y = row * panel_size + (row + 1) * padding + 25
        title_x = col * panel_size + (col + 1) * padding + panel_size // 2

        cv2.putText(canvas, pattern_name, (title_x - 50, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # 添加总标题
    cv2.putText(canvas, "Lesion Pattern Visualization Board", (canvas_width // 2 - 250, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    # 保存结果
    cv2.imwrite(output_path, canvas)
    print(f"SUCCESS: Visualization board saved: {output_path}")
    print(f"SIZE: {canvas_width}x{canvas_height}")


def batch_process_patterns(image_dir, mask_dir, output_dir='pattern_boards'):
    """
    批量处理目录中的图像对

    Parameters:
    -----------
    image_dir : str
        图像目录
    mask_dir : str
        mask目录
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取图像文件列表
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    img_files = [f for f in os.listdir(image_dir)
                if f.lower().endswith(image_extensions)]

    if not img_files:
        print(f"ERROR: No image files found in {image_dir}")
        return

    print(f"INFO: Found {len(img_files)} images")

    # 限制处理数量，避免过多
    max_images = min(len(img_files), 5)
    img_files = img_files[:max_images]

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, img_file)

        if not os.path.exists(mask_path):
            print(f"WARNING: Skip {img_file} (mask not found)")
            continue

        output_path = os.path.join(output_dir, f'pattern_board_{img_file}')
        print(f"Processing {i+1}/{len(img_files)}: {img_file}")

        create_pattern_visualization_board(img_path, mask_path, output_path)

    print(f"COMPLETED: Batch processing finished! Results saved in: {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='简单病变模式可视化')
    parser.add_argument('--image', type=str,
                       default='D:\\HUBU.zhan\\GDM\\VM-UNet-main\\ph2_dataset\\split_data_processed\\train\\image\\IMD002.png',
                       help='输入图像路径')
    parser.add_argument('--mask', type=str,
                       default='D:\\HUBU.zhan\\GDM\\VM-UNet-main\\ph2_dataset\\split_data_processed\\train\\mask\\IMD002.png',
                       help='mask路径')
    parser.add_argument('--output', type=str, default='lesion_pattern_board.png',
                       help='输出图像路径')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')
    parser.add_argument('--img_dir', type=str,
                       default='D:\\HUBU.zhan\\GDM\\VM-UNet-main\\ph2_dataset\\split_data_processed\\train\\image',
                       help='批量处理图像目录')
    parser.add_argument('--mask_dir', type=str,
                       default='D:\\HUBU.zhan\\GDM\\VM-UNet-main\\ph2_dataset\\split_data_processed\\train\\mask',
                       help='批量处理mask目录')

    args = parser.parse_args()

    if args.batch:
        # 批量处理模式
        batch_process_patterns(args.img_dir, args.mask_dir)
    else:
        # 单图模式
        create_pattern_visualization_board(args.image, args.mask, args.output)