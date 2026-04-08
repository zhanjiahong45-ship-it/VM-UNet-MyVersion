import os
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# 导入模型 (确保路径正确)
from models.vmunet.vmunetff import VMUNet

# ==========================================
# ⚙️ 配置参数
# ==========================================
MODEL_PATH = '/root/root/VM-UNet/results/vmunet_isic18_Friday_27_March_2026_21h_52m_35s/checkpoints/best.pth'
INPUT_DIR = 'inputs'
OUTPUT_DIR = 'diagnostic_report'
IMG_SIZE = 256
THRESHOLD = 0.5

# 实验 B：尝试不同的跳跃连接权重
ALPHA_LIST = [1.0, 5.0, 10.0, 20.0]

# ISIC18 专用标准化参数
ISIC18_TEST_MEAN = 149.034
ISIC18_TEST_STD = 32.022


# ==========================================
# 🛠️ 增强型可视化工具
# ==========================================
def preprocess(img_pil, target_size):
    img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
    img = np.array(img_pil, dtype=np.float32)
    img_norm = (img - ISIC18_TEST_MEAN) / ISIC18_TEST_STD
    mi, ma = np.min(img_norm), np.max(img_norm)
    img_final = ((img_norm - mi) / (ma - mi)) * 255.0 if ma > mi else img_norm
    return torch.from_numpy(img_final).permute(2, 0, 1).contiguous().unsqueeze(0).float()


def get_enhanced_heatmap(bottleneck_feat, original_img_pil):
    """
    实验 A 升级版：采用鲁棒分位数归一化，解决 Mamba 边界亮线导致的全蓝问题
    """
    shape = bottleneck_feat.shape
    if len(shape) == 4:
        feat = bottleneck_feat[0]
    elif len(shape) == 3:
        B, L, C = shape
        H = W = int(np.sqrt(L))
        feat = bottleneck_feat[0].reshape(H, W, C).permute(2, 0, 1)
    else:
        return None, None

    # 1. 计算通道平均激活
    heatmap = torch.mean(feat, dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)  # 丢弃负激活

    # --- 🌟 核心修复：鲁棒归一化 ---
    # 2. 自动检测并剔除 2% 的最亮像素（通常是 Mamba 的边界扫描干扰线）
    v_min = np.percentile(heatmap, 2)
    v_max = np.percentile(heatmap, 98)  # 使用 98 分位数替代绝对最大值

    if v_max > v_min:
        heatmap = (heatmap - v_min) / (v_max - v_min + 1e-8)

    heatmap = np.clip(heatmap, 0, 1)  # 强制截断到 0-1

    # 3. 幂律变换 (降低 Gamma 到 1.5，防止过黑)
    heatmap = np.power(heatmap, 1.5)

    # 4. 调整大小并生成伪彩色图
    heatmap_resized = cv2.resize(heatmap, original_img_pil.size)
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    # 5. 生成叠加图
    orig_cv = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_cv, 0.6, heatmap_color, 0.4, 0)

    return heatmap_color, overlay


# ==========================================
# 🚀 执行诊断
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = VMUNet(num_classes=1, depths=[2, 2, 9, 2], depths_decoder=[2, 2, 2, 1]).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到权重文件: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict({k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))},
                          strict=False)
    model.eval()

    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f"💡 请在 {INPUT_DIR} 放入病灶图片")
        return

    print(f"🚀 正在生成修复后的增强诊断报告...")

    with torch.no_grad():
        for name in images:
            img_path = os.path.join(INPUT_DIR, name)
            img_pil = Image.open(img_path).convert('RGB')
            tensor = preprocess(img_pil, IMG_SIZE).to(device)

            # --- 实验 A (修复边界干扰) ---
            _, bn, _ = model(tensor, alpha=1.0)
            h_map, overlay = get_enhanced_heatmap(bn, img_pil)

            if h_map is not None:
                orig_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                combined_a = np.hstack([orig_cv, h_map, overlay])
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_ExpA_Result.png"), combined_a)

            # --- 实验 B ---
            plt.figure(figsize=(20, 5))
            for i, a in enumerate(ALPHA_LIST):
                p, _, _ = model(tensor, alpha=a)
                p_map = cv2.resize(p.squeeze().cpu().numpy(), img_pil.size)
                mask = (p_map > THRESHOLD).astype(np.uint8) * 255
                plt.subplot(1, len(ALPHA_LIST), i + 1)
                plt.imshow(mask, cmap='gray')
                plt.title(f"Alpha (Detail Weight) = {a}")
                plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_ExpB_Comparison.png"))
            plt.close()

            print(f"✔ {name} 诊断修复成功")

    print(f"🎉 报告已更新至 '{OUTPUT_DIR}'，请重新查看 A 图，干扰线应该变淡，病灶特征会显现。")


if __name__ == '__main__':
    main()