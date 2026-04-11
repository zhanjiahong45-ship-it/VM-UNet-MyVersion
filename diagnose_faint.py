"""
Quick diagnostic: visualize raw probability maps on faint lesion images.

Usage:
    python diagnose_faint.py --ckpt_early checkpoints/latest_epoch3.pth \
                             --ckpt_best  checkpoints/best.pth \
                             --img_dir    ./inputs/ \
                             --out_dir    ./diagnosis/

If you don't have an epoch-3 checkpoint saved, just run with --ckpt_best only.
The script will save side-by-side: original | GT | raw probability heatmap.

The heatmap is the RAW sigmoid output (0.0 to 1.0) WITHOUT thresholding.
This tells us:
  - If probs are 0.0-0.1 on faint tissue → model can't see it (representation problem)
  - If probs are 0.2-0.4 on faint tissue → model sees it but isn't confident (calibration/threshold problem)
  - If probs are patchy → model is confused about boundaries (loss/augmentation problem)
"""

import torch
import numpy as np
import cv2
import os
import argparse
from PIL import Image

# Adjust these imports to match your project structure
import sys
sys.path.insert(0, '/root/root/VM-UNet')
from models.vmunet.vmunetff import VMUNet
from configs.config_setting import setting_config as config


def load_model(ckpt_path, device):
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None,  # don't load pretrained, we load our checkpoint
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # Handle both raw state_dict and wrapped checkpoint
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()
    return model


def preprocess(img_path, size=256):
    """Same preprocessing as your test_transformer minus the ToTensor."""
    MEAN = 149.034
    STD = 32.022
    img = Image.open(img_path).convert('RGB')
    original = np.array(img)
    img_resized = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32)
    arr = (arr - MEAN) / STD
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn) * 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, original


def make_heatmap(prob_map, original_size):
    """Convert probability map to colored heatmap overlaid on nothing (pure probabilities)."""
    # Resize prob map to original image size
    h, w = original_size[:2]
    prob_resized = cv2.resize(prob_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Create two visualizations:
    # 1. Raw probability as grayscale (0=black, 1=white)
    gray = (prob_resized * 255).astype(np.uint8)

    # 2. Jet colormap for better visibility
    jet = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    return gray, jet, prob_resized


def diagnose(ckpt_path, img_dir, out_dir, label, device):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(ckpt_path, device)

    img_paths = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    print(f"\n{'='*60}")
    print(f"  Diagnosing: {label}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Images: {len(img_paths)}")
    print(f"{'='*60}")

    for img_path in sorted(img_paths):
        name = os.path.basename(img_path)
        tensor, original = preprocess(img_path)
        tensor = tensor.to(device)

        with torch.no_grad():
            out = model(tensor)
            if isinstance(out, tuple):
                out = out[0]

        prob = out.squeeze().cpu().numpy()  # Raw sigmoid probabilities!

        gray, jet, prob_full = make_heatmap(prob, original.shape)

        # ---- Key statistics ----
        # Where the lesion likely is (center region, rough heuristic)
        h, w = prob_full.shape
        center = prob_full[h//4:3*h//4, w//4:3*w//4]

        print(f"\n  {name}:")
        print(f"    Global prob range: [{prob_full.min():.4f}, {prob_full.max():.4f}]")
        print(f"    Center region mean prob: {center.mean():.4f}")
        print(f"    Pixels > 0.5: {(prob_full > 0.5).sum()} / {prob_full.size} "
              f"({(prob_full > 0.5).mean()*100:.1f}%)")
        print(f"    Pixels > 0.3: {(prob_full > 0.3).sum()} / {prob_full.size} "
              f"({(prob_full > 0.3).mean()*100:.1f}%)")
        print(f"    Pixels > 0.1: {(prob_full > 0.1).sum()} / {prob_full.size} "
              f"({(prob_full > 0.1).mean()*100:.1f}%)")

        # ---- Save composite image ----
        # Resize original to match
        orig_resized = cv2.resize(
            cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
            (prob_full.shape[1], prob_full.shape[0])
        )

        # Overlay jet on original
        overlay = cv2.addWeighted(orig_resized, 0.5, jet, 0.5, 0)

        # Gray prob as 3-channel
        gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Stack: original | raw_prob_gray | jet_overlay
        composite = np.hstack([orig_resized, gray3, overlay])

        # Add probability scale bar
        cv2.putText(composite, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, "Raw Prob (0=black, 1=white)", (orig_resized.shape[1]+10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(composite, "Prob Overlay (Jet)", (2*orig_resized.shape[1]+10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        save_path = os.path.join(out_dir, f"{label}_{name}")
        cv2.imwrite(save_path, composite)
        print(f"    Saved: {save_path}")

    print(f"\n  Done. Check {out_dir}/\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_best', type=str, required=True,
                        help='Path to best checkpoint')
    parser.add_argument('--ckpt_early', type=str, default=None,
                        help='Path to early epoch checkpoint (epoch 3-4) if available')
    parser.add_argument('--img_dir', type=str, default='./inputs/',
                        help='Directory with faint lesion test images')
    parser.add_argument('--out_dir', type=str, default='./diagnosis/',
                        help='Where to save diagnostic images')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    diagnose(args.ckpt_best, args.img_dir, args.out_dir, "best_ckpt", device)

    if args.ckpt_early:
        diagnose(args.ckpt_early, args.img_dir, args.out_dir, "early_ckpt", device)
        print("="*60)
        print("  COMPARE the two sets of outputs!")
        print("  If early_ckpt shows higher probs on faint tissue → forgetting problem")
        print("  If both show near-zero probs → representation problem")
        print("="*60)