"""
Lesion Pattern Visualization Script
==================================
可视化多种病变衰减模式：弥漫、边界、中心、碎片等

基于现有代码中的faint_lesion_augmentor.py实现
展示不同模式的效果和模型预测结果
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import torch
from PIL import Image

# 导入模型和衰减器
from models.vmunet.vmunetff import VMUNet
from faint_lesion_augmentor import myFaintLesionAugmentor


class LesionPatternVisualizer:
    """病变模式可视化器"""

    def __init__(self, model_path, img_size=256, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size

        # 初始化模型
        self.model = VMUNet(
            num_classes=1,
            input_channels=3,
            depths=[2, 2, 9, 2],
            depths_decoder=[2, 2, 2, 1],
            drop_path_rate=0.2,
            load_ckpt_path=None
        ).to(self.device)

        # 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict') or checkpoint.get('model') or checkpoint
            # 清理统计参数
            state_dict = {k: v for k, v in state_dict.items()
                         if not k.endswith(('total_ops', 'total_params'))}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ 模型加载成功: {model_path}")
        else:
            print(f"⚠️  未找到模型文件: {model_path}")

        self.model.eval()

        # 初始化不同模式的衰减器
        self.augmentors = {
            'diffuse': myFaintLesionAugmentor(p=1.0, fade_range=(0.5, 0.8)),
            'boundary': myFaintLesionAugmentor(p=1.0, fade_range=(0.5, 0.8), mode_probs={'donut': 1.0}),
            'multifocal': myFaintLesionAugmentor(p=1.0, fade_range=(0.5, 0.8), mode_probs={'multi': 1.0}),
            'partial': myFaintLesionAugmentor(p=1.0, fade_range=(0.5, 0.8), mode_probs={'partial': 1.0}),
        }

        # 模式中文名称
        self.pattern_names_cn = {
            'original': '原始图像',
            'diffuse': '弥漫型衰减',
            'boundary': '边界型衰减',
            'multifocal': '多中心/碎片型衰减',
            'partial': '部分型衰减',
        }

    def preprocess_image(self, img_pil):
        """预处理图像"""
        img_pil = img_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = np.array(img_pil, dtype=np.float32)

        # ISIC18标准化
        ISIC18_MEAN = 149.034
        ISIC18_STD = 32.022
        img_normalized = (img - ISIC18_MEAN) / ISIC18_STD
        img_min, img_max = img_normalized.min(), img_normalized.max()

        if img_max > img_min:
            img_final = ((img_normalized - img_min) / (img_max - img_min)) * 255.0
        else:
            img_final = img_normalized

        img_tensor = torch.from_numpy(img_final).permute(2, 0, 1).contiguous().float()
        return img_tensor.unsqueeze(0)

    def predict(self, img_tensor):
        """模型预测"""
        with torch.no_grad():
            output = self.model(img_tensor.to(self.device))
            if isinstance(output, tuple):
                output = output[0]
            prob_map = output.squeeze().cpu().numpy()
        return prob_map

    def create_pattern_variants(self, image, mask):
        """创建不同模式的衰减变体"""
        variants = {'original': (image.copy(), mask.copy())}

        for pattern_name, augmentor in self.augmentors.items():
            aug_img, aug_mask = augmentor((image.copy(), mask.copy()))
            variants[pattern_name] = (aug_img, aug_mask)

        return variants

    def analyze_pattern_characteristics(self, mask, pattern_name):
        """分析不同模式的特征"""
        if mask.ndim == 3:
            mask_bin = (mask[:, :, 0] > 127).astype(np.uint8)
        else:
            mask_bin = (mask > 127).astype(np.uint8)

        # 计算基本特征
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {
                'area_ratio': 0,
                'num_contours': 0,
                'compactness': 0,
                'description': '无病变区域'
            }

        areas = [cv2.contourArea(c) for c in contours]
        total_area = sum(areas)

        # 计算轮廓数量（碎片化程度）
        num_contours = len(contours)

        # 计算紧致度 (圆形度)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(max_contour, True)
            compactness = 4 * np.pi * cv2.contourArea(max_contour) / (perimeter ** 2 + 1e-8)
        else:
            compactness = 0

        # 模式描述
        if pattern_name == 'diffuse':
            desc = "整体边界模糊，均匀衰减"
        elif pattern_name == 'boundary':
            desc = "边缘衰减明显，中心相对保留"
        elif pattern_name == 'multifocal':
            desc = f"分离成{num_contours}个独立病灶"
        elif pattern_name == 'partial':
            desc = "不对称部分衰减"
        else:
            desc = "原始病变形态"

        return {
            'area_ratio': total_area / (mask_bin.shape[0] * mask_bin.shape[1]),
            'num_contours': num_contours,
            'compactness': compactness,
            'description': desc
        }

    def visualize_single_image_patterns(self, img_path, mask_path=None, save_path=None):
        """可视化单个图像的不同模式"""
        # 读取图像
        img_pil = Image.open(img_path).convert('RGB')
        img_original = np.array(img_pil)

        # 读取或创建mask
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # 创建示例mask（中心圆形）
            h, w = img_original.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w//2, h//2), min(h, w)//4, 255, -1)

        # 创建不同模式变体
        variants = self.create_pattern_variants(img_original, mask)

        # 创建大图
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)

        pattern_order = ['original', 'diffuse', 'boundary', 'multifocal', 'partial']

        # 第一行：图像展示
        for idx, pattern_name in enumerate(pattern_order):
            ax = fig.add_subplot(gs[0, idx])
            aug_img, aug_mask = variants[pattern_name]

            # 显示图像
            ax.imshow(aug_img)
            ax.set_title(f'{self.pattern_names_cn[pattern_name]}', fontsize=14, fontweight='bold')

            # 叠加mask轮廓
            if aug_mask.ndim == 3:
                mask_vis = (aug_mask[:, :, 0] > 127).astype(np.uint8)
            else:
                mask_vis = (aug_mask > 127).astype(np.uint8)

            contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_points = contour.squeeze().reshape(-1, 2)
                ax.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)

            ax.axis('off')

        # 第二行：Mask展示和模型预测
        for idx, pattern_name in enumerate(pattern_order):
            ax = fig.add_subplot(gs[1, idx])
            aug_img, aug_mask = variants[pattern_name]

            # 模型预测
            img_pil_variant = Image.fromarray(aug_img.astype(np.uint8))
            img_tensor = self.preprocess_image(img_pil_variant)
            prob_map = self.predict(img_tensor)

            # 显示预测概率图
            im = ax.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
            ax.set_title('模型预测概率', fontsize=12)

            # 添加颜色条
            if idx == 4:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('预测概率', rotation=270, labelpad=15)

            ax.axis('off')

        # 第三行：特征分析
        for idx, pattern_name in enumerate(pattern_order):
            ax = fig.add_subplot(gs[2, idx])
            aug_img, aug_mask = variants[pattern_name]

            # 分析特征
            features = self.analyze_pattern_characteristics(aug_mask, pattern_name)

            # 创建特征文本
            feature_text = f"""特征分析:
━━━━━━━━━━━━━━━━
面积占比: {features['area_ratio']:.3f}
轮廓数量: {features['num_contours']}
紧致度: {features['compactness']:.3f}

{features['description']}
"""

            ax.text(0.1, 0.5, feature_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                   family='monospace')

            ax.axis('off')

        # 添加总标题
        fig.suptitle('病变衰减模式综合可视化分析', fontsize=18, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 可视化结果已保存: {save_path}")
        else:
            plt.show()

        plt.close()

    def create_comparison_dashboard(self, img_paths, mask_paths=None, save_path=None):
        """创建多图像对比仪表板"""
        n_images = min(len(img_paths), 4)  # 最多显示4张图像

        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(n_images, 6, figure=fig, hspace=0.2, wspace=0.2)

        pattern_order = ['original', 'diffuse', 'boundary', 'multifocal', 'partial']

        for img_idx in range(n_images):
            # 读取图像
            img_pil = Image.open(img_paths[img_idx]).convert('RGB')
            img_original = np.array(img_pil)

            # 读取mask
            if mask_paths and img_idx < len(mask_paths) and mask_paths[img_idx]:
                mask = cv2.imread(mask_paths[img_idx], cv2.IMREAD_GRAYSCALE)
            else:
                h, w = img_original.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (w//2, h//2), min(h, w)//4, 255, -1)

            # 创建变体
            variants = self.create_pattern_variants(img_original, mask)

            # 第一列：原始图像
            ax = fig.add_subplot(gs[img_idx, 0])
            ax.imshow(img_original)
            ax.set_title(f'图像 {img_idx + 1}: 原始', fontsize=12, fontweight='bold')
            if mask.ndim == 3:
                mask_vis = (mask[:, :, 0] > 127).astype(np.uint8)
            else:
                mask_vis = (mask > 127).astype(np.uint8)
            contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_points = contour.squeeze().reshape(-1, 2)
                ax.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=2)
            ax.axis('off')

            # 其他列：不同模式
            for col_idx, pattern_name in enumerate(pattern_order[1:], 1):
                ax = fig.add_subplot(gs[img_idx, col_idx])
                aug_img, aug_mask = variants[pattern_name]

                # 显示衰减后图像
                ax.imshow(aug_img)
                ax.set_title(self.pattern_names_cn[pattern_name], fontsize=10)

                # 叠加轮廓
                if aug_mask.ndim == 3:
                    mask_vis = (aug_mask[:, :, 0] > 127).astype(np.uint8)
                else:
                    mask_vis = (aug_mask > 127).astype(np.uint8)

                contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    contour_points = contour.squeeze().reshape(-1, 2)
                    ax.plot(contour_points[:, 0], contour_points[:, 1], 'r-', linewidth=1.5)

                ax.axis('off')

        fig.suptitle('多图像病变衰减模式对比分析', fontsize=20, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"✅ 对比仪表板已保存: {save_path}")
        else:
            plt.show()

        plt.close()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='病变衰减模式可视化')
    parser.add_argument('--model_path', type=str,
                       default='/root/root/VM-UNet/results/FC_8255/checkpoints/early_epochs/epoch_004_miou0.7906.pth',
                       help='模型权重路径')
    parser.add_argument('--input_dir', type=str, default='inputs',
                       help='输入图像目录')
    parser.add_argument('--mask_dir', type=str, default=None,
                       help='对应mask目录')
    parser.add_argument('--output_dir', type=str, default='pattern_visualization',
                       help='输出目录')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='batch',
                       help='可视化模式：single(单图详细分析) 或 batch(批量对比)')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化可视化器
    visualizer = LesionPatternVisualizer(args.model_path)

    # 获取图像列表
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(args.input_dir)
                if f.lower().endswith(image_extensions)]

    if not img_files:
        print(f"❌ 在 {args.input_dir} 中未找到图像文件")
        return

    print(f"📁 找到 {len(img_files)} 张图像")

    # 处理mask路径
    mask_files = []
    if args.mask_dir and os.path.exists(args.mask_dir):
        mask_files = [os.path.join(args.mask_dir, f) for f in img_files]
    img_files = [os.path.join(args.input_dir, f) for f in img_files]

    # 执行可视化
    if args.mode == 'single':
        # 单图详细分析模式
        for i, (img_path, mask_path) in enumerate(zip(img_files, mask_files or [None]*len(img_files))):
            save_path = os.path.join(args.output_dir, f'pattern_analysis_{i+1}.png')
            print(f"🔍 处理图像 {i+1}/{len(img_files)}: {os.path.basename(img_path)}")
            visualizer.visualize_single_image_patterns(img_path, mask_path, save_path)

    else:
        # 批量对比模式
        save_path = os.path.join(args.output_dir, 'comparison_dashboard.png')
        print("📊 创建批量对比仪表板...")
        visualizer.create_comparison_dashboard(img_files[:4], mask_files[:4] if mask_files else None, save_path)

    print(f"🎉 可视化完成！结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()