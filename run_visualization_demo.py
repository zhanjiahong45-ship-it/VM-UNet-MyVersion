


"""
快速演示脚本 - 病变模式可视化
================================
简单易用的演示，展示4种病变衰减模式
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from faint_lesion_augmentor import myFaintLesionAugmentor


def quick_pattern_demo(image_path, mask_path=None, output_path='pattern_demo.png'):
    """
    快速演示：展示4种病变衰减模式

    Parameters:
    -----------
    image_path : str
        原始图像路径
    mask_path : str, optional
        病变mask路径，如不提供则创建示例mask
    output_path : str
        输出图像路径
    """

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 读取或创建mask
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    else:
        # 创建示例mask（中心椭圆形病变）
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w//2, h//2), (w//4, h//4), 0, 0, 360, 255, -1)
        print("ℹ️  使用示例mask（中心椭圆）")

    # 创建不同模式的衰减器
    patterns = {
        '原始图像': None,
        '弥漫型衰减': myFaintLesionAugmentor(p=1.0, mode_probs={'diffuse': 1.0}),
        '边界型衰减': myFaintLesionAugmentor(p=1.0, mode_probs={'donut': 1.0}),
        '多中心型衰减': myFaintLesionAugmentor(p=1.0, mode_probs={'multi': 1.0}),
        '部分型衰减': myFaintLesionAugmentor(p=1.0, mode_probs={'partial': 1.0}),
    }

    # 创建画布
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('病变衰减模式演示', fontsize=20, fontweight='bold')

    # 生成不同模式变体
    for idx, (pattern_name, augmentor) in enumerate(patterns.items()):
        ax = axes[idx // 3, idx % 3]

        if augmentor is None:
            # 原始图像
            result_img = img.copy()
            result_mask = mask.copy()
        else:
            # 应用衰减
            result_img, result_mask = augmentor((img.copy(), mask.copy()))

        # 显示图像
        ax.imshow(result_img)

        # 叠加mask轮廓
        if result_mask.ndim == 3:
            mask_vis = (result_mask[:, :, 0] > 127).astype(np.uint8)
        else:
            mask_vis = (result_mask > 127).astype(np.uint8)

        contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_points = contour.squeeze().reshape(-1, 2)
            ax.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)

        ax.set_title(pattern_name, fontsize=14, fontweight='bold')
        ax.axis('off')

    # 隐藏最后一个子图
    axes[1, 2].axis('off')
    axes[1, 2].text(0.5, 0.5, '基于\nfaint_lesion_augmentor.py\n实现',
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 演示图像已保存: {output_path}")
    plt.close()


def batch_process_directory(input_dir, output_dir='pattern_outputs'):
    """
    批量处理目录中的所有图像

    Parameters:
    -----------
    input_dir : str
        输入图像目录
    output_dir : str
        输出目录
    """

    os.makedirs(output_dir, exist_ok=True)

    # 支持的图像格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(input_dir)
                if f.lower().endswith(image_extensions)]

    if not img_files:
        print(f"❌ 在 {input_dir} 中未找到图像文件")
        return

    print(f"📁 找到 {len(img_files)} 张图像")

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f'pattern_{img_file}')

        print(f"🔍 处理 {i+1}/{len(img_files)}: {img_file}")
        quick_pattern_demo(img_path, output_path=output_path)

    print(f"🎉 批量处理完成！结果保存在: {output_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='病变模式快速演示')
    parser.add_argument('--input', type=str, default='inputs',
                       help='输入图像路径或目录')
    parser.add_argument('--mask', type=str, default=None,
                       help='mask路径（可选）')
    parser.add_argument('--output', type=str, default='pattern_demo.png',
                       help='输出图像路径或目录')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')

    args = parser.parse_args()

    if args.batch:
        # 批量处理模式
        batch_process_directory(args.input, args.output)
    else:
        # 单图演示模式
        quick_pattern_demo(args.input, args.mask, args.output)