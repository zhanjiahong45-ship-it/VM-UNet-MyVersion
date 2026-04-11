import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# 导入你的模型
from models.vmunet.vmunetff import VMUNet


def get_isic_transform(img_size=256):
    """手动实现 ISIC18 的预处理，与测试集完全一致"""
    ISIC18_TEST_MEAN = 149.034
    ISIC18_TEST_STD = 32.022

    def transform(img_pil):
        # 1. Resize
        img_resized = img_pil.resize((img_size, img_size), Image.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32)

        # 2. Normalize
        img_normalized = (img_arr - ISIC18_TEST_MEAN) / ISIC18_TEST_STD

        # 3. Min-Max Scale to 0-255 (你的原始代码逻辑)
        img_min, img_max = np.min(img_normalized), np.max(img_normalized)
        if img_max > img_min:
            img_final = ((img_normalized - img_min) / (img_max - img_min)) * 255.0
        else:
            img_final = img_normalized

        # 4. ToTensor: HWC -> CHW
        img_tensor = torch.from_numpy(img_final).permute(2, 0, 1).contiguous().float()
        return img_tensor.unsqueeze(0)  # 添加 batch 维度

    return transform


def main():
    # ================= 配置区域 =================
    # 1. 填入你跑出的 best 权重路径
    ckpt_path = '/root/root/VM-UNet/results/vmunet_isic18_Thursday_09_April_2026_22h_29m_21s/checkpoints/best.pth'

    # 2. 填入你要测试的那三张困难图片的绝对或相对路径
    img_paths = [
        '/root/root/VM-UNet/inputs/3.png',
        '/root/root/VM-UNet/inputs/img_1.png',
        '/root/root/VM-UNet/inputs/img_2.png'
    ]

    # 3. 设定二值化阈值 (可以改改看，比如 0.3 或 0.5)
    THRESHOLD = 0.3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ============================================

    print("=> 正在加载模型...")
    model = VMUNet(
        num_classes=1,
        input_channels=3,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 2, 2, 1],
        drop_path_rate=0.2,
        load_ckpt_path=None  # 推理时不需要预训练的 VMamba 权重了
    )

    # 加载权重
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        # 兼容处理：如果你存的是完整的 dict 还是只有 state_dict
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("=> 权重加载成功！")
    except Exception as e:
        print(f"=> 权重加载失败，请检查路径: {e}")
        return

    model = model.to(device)
    model.eval()

    transform = get_isic_transform(256)

    # 创建一个画板 (3行，每行3列: 原图, 概率热力图, 预测Mask)
    fig, axes = plt.subplots(len(img_paths), 3, figsize=(15, 5 * len(img_paths)))
    if len(img_paths) == 1:
        axes = [axes]

    print("=> 开始推理与生成热力图...")
    with torch.no_grad():
        for i, path in enumerate(img_paths):
            if not os.path.exists(path):
                print(f"找不到图片: {path}")
                continue

            # 1. 读取原图
            img_pil = Image.open(path).convert('RGB')
            img_tensor = transform(img_pil).to(device)

            # 2. 模型前向传播
            output = model(img_tensor)
            if isinstance(output, tuple):
                output = output[0]

            # 3. 获取概率图 (Probability Map) [0, 1] 之间
            prob_map = output.squeeze().cpu().numpy()

            # 4. 生成二值化预测 (Binary Mask)
            pred_mask = (prob_map > THRESHOLD).astype(np.uint8) * 255

            # ----- 开始绘图 -----
            ax_orig, ax_prob, ax_pred = axes[i]

            # 画原图
            ax_orig.imshow(img_pil)
            ax_orig.set_title(f"Image {i + 1}")
            ax_orig.axis('off')

            # 画概率热力图 (最核心的一步！)
            # 使用 jet colormap: 红色表示接近 1 (肯定是病灶)，蓝色表示接近 0 (肯定是皮肤)
            im = ax_prob.imshow(prob_map, cmap='jet', vmin=0, vmax=1)
            ax_prob.set_title(f"Prob Map (Min:{prob_map.min():.2f}, Max:{prob_map.max():.2f})")
            ax_prob.axis('off')
            fig.colorbar(im, ax=ax_prob, fraction=0.046, pad=0.04)  # 加上颜色条

            # 画二值化 Mask
            ax_pred.imshow(pred_mask, cmap='gray')
            ax_pred.set_title(f"Predicted Mask (Thresh > {THRESHOLD})")
            ax_pred.axis('off')

    plt.tight_layout()
    save_path = "prob_visualization.png"
    plt.savefig(save_path, dpi=200)
    print(f"=> 完成！可视化结果已保存至: {save_path}")


if __name__ == '__main__':
    main()