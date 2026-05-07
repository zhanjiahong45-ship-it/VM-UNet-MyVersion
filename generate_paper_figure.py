"""
生成适合Springer论文的横向病变模式图
=======================================
创建1x5布局的高质量图像，适合直接插入论文
"""

import os
import cv2
import numpy as np
from faint_lesion_augmentor import myFaintLesionAugmentor


def create_springer_paper_figure(image_path, mask_path, output_path='paper_figure.png', dpi=300):
    """
    创建适合Springer论文的横向布局图

    Parameters:
    -----------
    image_path : str
        原始图像路径
    mask_path : str
        病变mask路径
    output_path : str
        输出图像路径
    dpi : int
        分辨率，默认300适合论文出版
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
        print("INFO: Using example mask")

    # 确保mask是二值的
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 创建不同模式的衰减器
    patterns = [
        ('Original', None),
        ('Diffuse', myFaintLesionAugmentor(p=1.0, mode_probs={'diffuse': 1.0})),
        ('Boundary', myFaintLesionAugmentor(p=1.0, mode_probs={'donut': 1.0})),
        ('Multifocal', myFaintLesionAugmentor(p=1.0, mode_probs={'multi': 1.0})),
        ('Partial', myFaintLesionAugmentor(p=1.0, mode_probs={'partial': 1.0})),
    ]

    # 设置适合论文的尺寸
    # Springer建议：单栏图宽度通常为85mm（约3.35英寸），双栏图宽度为174mm（约6.85英寸）
    panel_height = 400  # 每个面板高度
    panel_width = 400   # 每个面板宽度
    padding = 30        # 面板间距
    label_height = 60   # 标签高度

    # 计算整个画布的尺寸 (1行5列)
    canvas_width = 5 * panel_width + 6 * padding
    canvas_height = panel_height + 2 * padding + label_height

    # 创建白色背景画布
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # 生成不同模式变体
    for idx, (pattern_name, augmentor) in enumerate(patterns):
        # 计算面板位置
        x_start = idx * panel_width + (idx + 1) * padding
        y_start = padding + label_height

        if augmentor is None:
            # 原始图像
            result_img = img.copy()
            result_mask = mask_binary.copy()
        else:
            # 应用衰减
            result_img, result_mask = augmentor((img.copy(), mask_binary.copy()))

        # 调整图像大小以适应面板
        img_resized = cv2.resize(result_img, (panel_width, panel_height))

        # 叠加mask轮廓
        if result_mask.ndim == 3:
            mask_vis = (result_mask[:, :, 0] > 127).astype(np.uint8)
        else:
            mask_vis = (result_mask > 127).astype(np.uint8)

        # 在调整大小前找到轮廓
        contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 调整contour坐标以适应缩放后的图像
        scale_x = panel_width / result_mask.shape[1]
        scale_y = panel_height / result_mask.shape[0]

        # 绘制图像
        canvas[y_start:y_start+panel_height, x_start:x_start+panel_width] = img_resized

        # 绘制轮廓 - 使用更明显的颜色和线宽适合论文
        for contour in contours:
            contour_scaled = contour.copy()
            contour_scaled[:, 0, 0] = (contour[:, 0, 0] * scale_x + x_start).astype(int)
            contour_scaled[:, 0, 1] = (contour[:, 0, 1] * scale_y + y_start).astype(int)
            # 使用红色轮廓，线宽3适合论文
            cv2.drawContours(canvas, [contour_scaled], -1, (0, 0, 255), 3)

        # 添加子图标签 (a), (b), (c), (d), (e)
        label_char = chr(ord('a') + idx)
        label_x = x_start + 20
        label_y = y_start - 10

        # 子图标签
        cv2.putText(canvas, f"({label_char})", (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

        # 添加模式名称
        title_y = y_start + panel_height + 40
        title_x = x_start + panel_width // 2 - len(pattern_name) * 15

        cv2.putText(canvas, pattern_name, (title_x, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # 添加总标题（可选，论文中通常在图注中说明）
    # title_text = "Lesion Pattern Variations"
    # cv2.putText(canvas, title_text, (canvas_width // 2 - 200, 40),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)

    # 保存结果 - 使用高质量PNG
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"SUCCESS: Paper figure saved: {output_path}")
    print(f"SIZE: {canvas_width}x{canvas_height} pixels")
    print(f"ESTIMATED PRINT SIZE: {canvas_width/dpi*2.54:.1f}cm x {canvas_height/dpi*2.54:.1f}cm @ {dpi}dpi")

    return canvas


def create_multi_row_figure(image_paths, mask_paths, output_path='paper_figure_multirow.png', dpi=300):
    """
    创建多行布局的论文图像（如果有多个样本需要展示）

    Parameters:
    -----------
    image_paths : list
        图像路径列表
    mask_paths : list
        mask路径列表
    output_path : str
        输出图像路径
    dpi : int
        分辨率
    """

    panel_height = 350
    panel_width = 350
    padding = 25
    label_height = 50

    num_rows = len(image_paths)
    num_cols = 5  # Original + 4 patterns

    canvas_width = num_cols * panel_width + (num_cols + 1) * padding
    canvas_height = num_rows * panel_height + (num_rows + 1) * padding + label_height

    # 创建白色背景画布
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    patterns = [
        ('Original', None),
        ('Diffuse', myFaintLesionAugmentor(p=1.0, mode_probs={'diffuse': 1.0})),
        ('Boundary', myFaintLesionAugmentor(p=1.0, mode_probs={'donut': 1.0})),
        ('Multifocal', myFaintLesionAugmentor(p=1.0, mode_probs={'multi': 1.0})),
        ('Partial', myFaintLesionAugmentor(p=1.0, mode_probs={'partial': 1.0})),
    ]

    for row_idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print(f"WARNING: Skip row {row_idx+1} - cannot read image/mask")
            continue

        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        for col_idx, (pattern_name, augmentor) in enumerate(patterns):
            x_start = col_idx * panel_width + (col_idx + 1) * padding
            y_start = row_idx * panel_height + (row_idx + 1) * padding + label_height

            if augmentor is None:
                result_img = img.copy()
                result_mask = mask_binary.copy()
            else:
                result_img, result_mask = augmentor((img.copy(), mask_binary.copy()))

            img_resized = cv2.resize(result_img, (panel_width, panel_height))

            if result_mask.ndim == 3:
                mask_vis = (result_mask[:, :, 0] > 127).astype(np.uint8)
            else:
                mask_vis = (result_mask > 127).astype(np.uint8)

            contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            scale_x = panel_width / result_mask.shape[1]
            scale_y = panel_height / result_mask.shape[0]

            canvas[y_start:y_start+panel_height, x_start:x_start+panel_width] = img_resized

            for contour in contours:
                contour_scaled = contour.copy()
                contour_scaled[:, 0, 0] = (contour[:, 0, 0] * scale_x + x_start).astype(int)
                contour_scaled[:, 0, 1] = (contour[:, 0, 1] * scale_y + y_start).astype(int)
                cv2.drawContours(canvas, [contour_scaled], -1, (0, 0, 255), 2)

            # 只在第一行添加标签
            if row_idx == 0:
                label_char = chr(ord('a') + col_idx)
                label_x = x_start + 15
                label_y = y_start - 10
                cv2.putText(canvas, f"({label_char})", (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

            # 只在第一行添加标题
            if row_idx == 0:
                title_y = y_start + panel_height + 35
                title_x = x_start + panel_width // 2 - len(pattern_name) * 12
                cv2.putText(canvas, pattern_name, (title_x, title_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # 保存结果
    cv2.imwrite(output_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"SUCCESS: Multi-row paper figure saved: {output_path}")
    print(f"SIZE: {canvas_width}x{canvas_height} pixels ({num_rows} rows x {num_cols} columns)")

    return canvas


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate Springer paper figures')
    parser.add_argument('--image', type=str,
                       default='D:\\HUBU.zhan\\GDM\\VM-UNet-main\\ph2_dataset\\split_data_processed\\train\\image\\IMD002.png',
                       help='Input image path')
    parser.add_argument('--mask', type=str,
                       default='D:\\HUBU.zhan\\GDM\\VM-UNet-main\\ph2_dataset\\split_data_processed\\train\\mask\\IMD002.png',
                       help='Mask path')
    parser.add_argument('--output', type=str, default='springer_paper_figure.png',
                       help='Output image path')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output resolution (default: 300 for publication)')
    parser.add_argument('--multirow', action='store_true',
                       help='Generate multi-row figure')
    parser.add_argument('--img_dir', type=str,
                       default='D:\\HUBU.zhan\\GDM\\VM-UNet-main\\ph2_dataset\\split_data_processed\\train\\image',
                       help='Image directory for multi-row mode')
    parser.add_argument('--mask_dir', type=str,
                       default='D:\\HUBU.zhan\\GDM\\VM-UNet-main\\ph2_dataset\\split_data_processed\\train\\mask',
                       help='Mask directory for multi-row mode')
    parser.add_argument('--num_rows', type=int, default=2,
                       help='Number of rows for multi-row mode')

    args = parser.parse_args()

    if args.multirow:
        # 多行模式
        import os
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        img_files = [f for f in os.listdir(args.img_dir)
                    if f.lower().endswith(image_extensions)][:args.num_rows]

        image_paths = [os.path.join(args.img_dir, f) for f in img_files]
        mask_paths = [os.path.join(args.mask_dir, f) for f in img_files]

        create_multi_row_figure(image_paths, mask_paths, args.output, args.dpi)
    else:
        # 单行模式（默认）
        create_springer_paper_figure(args.image, args.mask, args.output, args.dpi)