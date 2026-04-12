"""
独立可视化脚本：对 inputs/ 目录下的图片运行双模型7格面板并保存。
使用前修改下方 ===用户配置=== 里的路径。
"""

import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# ===用户配置===================================================
MIXTURE_CKPT    = r'/root/root/VM-UNet/results/vmunet_isic18_Sunday_12_April_2026_20h_04m_13s/checkpoints/mixture.pth'      # mixture.pth 路径
INPUT_DIR       = r'inputs'                   # 待可视化图片目录
OUTPUT_DIR      = r'vis_output'               # 结果保存目录
INPUT_SIZE      = (256, 256)                  # (H, W)，和训练时一致
FAINT_THRESHOLD = 0.4                         # inpaint 阈值
# =============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 导入模型 ──────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models.vmunet.vmunetff import VMUNet
from configs.config_setting import setting_config

config = setting_config


def build_model(device):
    m = VMUNet(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        depths=config.model_config['depths'],
        depths_decoder=config.model_config['depths_decoder'],
        drop_path_rate=config.model_config['drop_path_rate'],
        load_ckpt_path=None,
    ).to(device)
    return m


def load_mixture(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    early_model = build_model(device)
    early_model.load_state_dict(ckpt['early_model'], strict=False)
    early_model.eval()
    return model, early_model


def inpaint_with_skin(img_tensor, prob, threshold=0.4):
    anchor = (prob > threshold).float()
    non_anchor = 1.0 - anchor
    skin_color = (img_tensor * non_anchor).sum(dim=[2, 3], keepdim=True) / \
                 (non_anchor.sum(dim=[2, 3], keepdim=True) + 1e-8)
    return img_tensor * (1 - anchor) + skin_color.expand_as(img_tensor) * anchor


def tensor_to_display(t):
    """把任意值域的 [1,C,H,W] tensor 转成可 imshow 的 [H,W,3] numpy，归一化到 [0,1]。"""
    arr = t.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def preprocess(img_path, device):
    """读图 → resize → 转 tensor [1,C,H,W]，值域与训练归一化一致。"""
    H, W = INPUT_SIZE
    img = np.array(Image.open(img_path).convert('RGB').resize((W, H)))
    dummy_mask = np.zeros((H, W, 1), dtype=np.float32)
    img_t, _ = config.test_transformer((img, dummy_mask))
    return img, img_t.unsqueeze(0).float().to(device)


def run(img_path, model, early_model, device):
    img_name = os.path.basename(img_path)
    img_np, img_tensor = preprocess(img_path, device)

    with torch.no_grad():
        # ── 推理 ──────────────────────────────────────────────
        out_m = model(img_tensor)
        out_m = out_m[0] if isinstance(out_m, tuple) else out_m

        prob_max  = out_m.max().item()
        prob_mean = out_m.mean().item()
        is_faint  = prob_max > 0.5 and prob_mean < 0.05
        print(f"[{img_name}] max={prob_max:.3f} mean={prob_mean:.4f} faint={is_faint}")

        # ── 补色图 ────────────────────────────────────────────
        # 用主模型 mask 补色
        img_m1 = inpaint_with_skin(img_tensor, out_m, threshold=FAINT_THRESHOLD)
        # 用早期模型 mask 补色
        out_e = early_model(img_tensor)
        out_e = out_e[0] if isinstance(out_e, tuple) else out_e
        img_me = inpaint_with_skin(img_tensor, out_e, threshold=FAINT_THRESHOLD)

        # 早期模型跑补色图
        out_m1 = early_model(img_m1)
        out_m1 = out_m1[0] if isinstance(out_m1, tuple) else out_m1
        out_me = early_model(img_me)
        out_me = out_me[0] if isinstance(out_me, tuple) else out_me

    # ── 绘图 ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 7, figsize=(35, 5))
    fig.suptitle(f"{img_name}  max={prob_max:.3f} mean={prob_mean:.4f} faint={is_faint}", fontsize=11)

    axes[0].imshow(img_np)
    axes[0].set_title("1.Original"); axes[0].axis('off')

    axes[1].imshow(out_m.squeeze().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    axes[1].set_title(f"2.Mask-m(main)\nmax={prob_max:.2f}"); axes[1].axis('off')

    axes[2].imshow(out_e.squeeze().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    axes[2].set_title(f"3.Mask-e(early)\nmax={out_e.max().item():.2f}"); axes[2].axis('off')

    # 4、5 用归一化显示，确保看得见
    axes[3].imshow(tensor_to_display(img_m1))
    axes[3].set_title("4.Inpaint by Mask-m"); axes[3].axis('off')

    axes[4].imshow(tensor_to_display(img_me))
    axes[4].set_title("5.Inpaint by Mask-e"); axes[4].axis('off')

    axes[5].imshow(out_m1.squeeze().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    axes[5].set_title(f"6.early(inpaint-m)\nmax={out_m1.max().item():.2f}"); axes[5].axis('off')

    axes[6].imshow(out_me.squeeze().cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    axes[6].set_title(f"7.early(inpaint-e)\nmax={out_me.max().item():.2f}"); axes[6].axis('off')

    save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_vis.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  saved → {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    print(f"Loading mixture.pth from: {MIXTURE_CKPT}")
    model, early_model = load_mixture(MIXTURE_CKPT, device)

    img_paths = [
        os.path.join(INPUT_DIR, f)
        for f in sorted(os.listdir(INPUT_DIR))
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"Found {len(img_paths)} images in {INPUT_DIR}")

    for p in img_paths:
        run(p, model, early_model, device)

    print("Done.")


if __name__ == '__main__':
    main()
