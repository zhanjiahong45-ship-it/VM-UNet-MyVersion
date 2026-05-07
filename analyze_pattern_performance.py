"""
病变模式性能统计分析脚本
================================
统计分析不同衰减模式对模型预测性能的影响
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from faint_lesion_augmentor import myFaintLesionAugmentor
import torch
from PIL import Image
from sklearn.metrics import jaccard_score, accuracy_score, f1_score


class PatternPerformanceAnalyzer:
    """病变模式性能分析器"""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 导入模型
        from models.vmunet.vmunetff import VMUNet
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
            state_dict = {k: v for k, v in state_dict.items()
                         if not k.endswith(('total_ops', 'total_params'))}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ 模型加载成功")
        self.model.eval()

        # 创建不同模式的衰减器
        self.patterns = {
            'original': None,
            'diffuse': myFaintLesionAugmentor(p=1.0, mode_probs={'diffuse': 1.0}),
            'boundary': myFaintLesionAugmentor(p=1.0, mode_probs={'donut': 1.0}),
            'multifocal': myFaintLesionAugmentor(p=1.0, mode_probs={'multi': 1.0}),
            'partial': myFaintLesionAugmentor(p=1.0, mode_probs={'partial': 1.0}),
        }

        self.pattern_names_cn = {
            'original': '原始图像',
            'diffuse': '弥漫型',
            'boundary': '边界型',
            'multifocal': '多中心型',
            'partial': '部分型',
        }

    def preprocess_image(self, img_pil, target_size=256):
        """预处理图像"""
        img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
        img = np.array(img_pil, dtype=np.float32)

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

    def calculate_metrics(self, pred, gt_mask, threshold=0.5):
        """计算评估指标"""
        pred_binary = (pred > threshold).astype(np.uint8).flatten()
        gt_binary = (gt_mask > 127).astype(np.uint8).flatten()

        # 避免除零错误
        if gt_binary.sum() == 0:
            return {
                'iou': 0.0,
                'dice': 0.0,
                'accuracy': accuracy_score(gt_binary, pred_binary),
                'f1': 0.0,
            }

        iou = jaccard_score(gt_binary, pred_binary, zero_division=0)
        dice = f1_score(gt_binary, pred_binary, zero_division=0)
        accuracy = accuracy_score(gt_binary, pred_binary)
        f1 = f1_score(gt_binary, pred_binary, zero_division=0)

        return {
            'iou': iou,
            'dice': dice,
            'accuracy': accuracy,
            'f1': f1,
        }

    def analyze_dataset(self, img_dir, mask_dir=None, max_samples=20):
        """分析整个数据集"""
        # 获取图像列表
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        img_files = [f for f in os.listdir(img_dir)
                    if f.lower().endswith(image_extensions)]

        img_files = img_files[:max_samples]  # 限制样本数量
        print(f"📁 分析 {len(img_files)} 张图像")

        # 存储结果
        results = defaultdict(lambda: defaultdict(list))

        for i, img_file in enumerate(img_files):
            img_path = os.path.join(img_dir, img_file)

            # 读取图像
            img_pil = Image.open(img_path).convert('RGB')
            img_original = np.array(img_pil)

            # 读取mask
            if mask_dir:
                mask_path = os.path.join(mask_dir, img_file)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                else:
                    # 创建示例mask
                    h, w = img_original.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.circle(mask, (w//2, h//2), min(h, w)//4, 255, -1)
            else:
                h, w = img_original.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (w//2, h//2), min(h, w)//4, 255, -1)

            # 测试每种模式
            for pattern_name, augmentor in self.patterns.items():
                if augmentor is None:
                    aug_img = img_original.copy()
                else:
                    aug_img, _ = augmentor((img_original.copy(), mask.copy()))

                # 预测
                img_pil_aug = Image.fromarray(aug_img.astype(np.uint8))
                img_tensor = self.preprocess_image(img_pil_aug)
                prob_map = self.predict(img_tensor)

                # 调整预测结果大小以匹配原始mask
                prob_map_resized = cv2.resize(prob_map, (mask.shape[1], mask.shape[0]))

                # 计算指标
                metrics = self.calculate_metrics(prob_map_resized, mask)

                for metric_name, metric_value in metrics.items():
                    results[pattern_name][metric_name].append(metric_value)

            print(f"✓ 处理进度: {i+1}/{len(img_files)}")

        return results

    def create_performance_report(self, results, save_path='pattern_performance_report.png'):
        """创建性能报告"""
        # 计算统计信息
        summary = {}
        for pattern_name in self.patterns.keys():
            pattern_results = results[pattern_name]
            summary[pattern_name] = {}
            for metric_name, values in pattern_results.items():
                summary[pattern_name][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }

        # 创建DataFrame
        df_data = []
        for pattern_name in self.patterns.keys():
            row = {'模式': self.pattern_names_cn[pattern_name]}
            for metric_name in ['iou', 'dice', 'accuracy']:
                row[f'{metric_name.upper()}_均值'] = summary[pattern_name][metric_name]['mean']
                row[f'{metric_name.upper()}_标准差'] = summary[pattern_name][metric_name]['std']
            df_data.append(row)

        df = pd.DataFrame(df_data)

        # 创建可视化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('病变衰减模式性能分析报告', fontsize=16, fontweight='bold')

        metrics = ['iou', 'dice', 'accuracy']
        metric_names_cn = {'iou': 'IoU', 'dice': 'Dice系数', 'accuracy': '准确率'}

        # 绘制柱状图
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            patterns_cn = [self.pattern_names_cn[p] for p in self.patterns.keys()]
            means = [summary[p][metric]['mean'] for p in self.patterns.keys()]
            stds = [summary[p][metric]['std'] for p in self.patterns.keys()]

            bars = ax.bar(patterns_cn, means, yerr=stds, capsize=5, alpha=0.7,
                         color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

            # 添加数值标签
            for i, (bar, mean) in enumerate(zip(bars, means)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

            ax.set_ylabel(f'{metric_names_cn[metric]}', fontsize=12)
            ax.set_title(f'{metric_names_cn[metric]}对比', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.0])

        # 绘制性能下降率
        ax = axes[1, 1]
        original_iou = summary['original']['iou']['mean']

        degradation_data = []
        degradation_labels = []

        for pattern_name in ['diffuse', 'boundary', 'multifocal', 'partial']:
            pattern_iou = summary[pattern_name]['iou']['mean']
            degradation = (original_iou - pattern_iou) / original_iou * 100
            degradation_data.append(degradation)
            degradation_labels.append(self.pattern_names_cn[pattern_name])

        colors = ['#ff7f0e' if d > 0 else '#2ca02c' for d in degradation_data]
        bars = ax.bar(degradation_labels, degradation_data, color=colors, alpha=0.7)

        for bar, value in zip(bars, degradation_data):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%', ha='center', va='bottom' if value > 0 else 'top', fontsize=10)

        ax.set_ylabel('IoU下降率 (%)', fontsize=12)
        ax.set_title('相对原始图像的性能下降', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 性能报告已保存: {save_path}")
        plt.close()

        # 打印统计表格
        print("\n" + "="*80)
        print("性能统计摘要")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)

        return df, summary


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='病变模式性能分析')
    parser.add_argument('--model_path', type=str,
                       default='/root/root/VM-UNet/results/FC_8255/checkpoints/early_epochs/epoch_004_miou0.7906.pth',
                       help='模型权重路径')
    parser.add_argument('--img_dir', type=str, default='inputs',
                       help='测试图像目录')
    parser.add_argument('--mask_dir', type=str, default=None,
                       help='对应mask目录')
    parser.add_argument('--output', type=str, default='pattern_performance_report.png',
                       help='输出报告路径')
    parser.add_argument('--max_samples', type=int, default=20,
                       help='最大分析样本数')

    args = parser.parse_args()

    # 初始化分析器
    analyzer = PatternPerformanceAnalyzer(args.model_path)

    # 分析数据集
    results = analyzer.analyze_dataset(args.img_dir, args.mask_dir, args.max_samples)

    # 生成报告
    df, summary = analyzer.create_performance_report(results, args.output)

    print(f"\n🎉 分析完成！报告已保存: {args.output}")


if __name__ == '__main__':
    main()