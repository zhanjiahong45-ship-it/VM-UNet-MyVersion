"""
Visualize synthetic faint-lesion augmentation.

This takes DARK lesion images (with their GT masks) and fades them
to simulate faint lesions. Run on a few training images to verify.

Usage:
    python test_faint_sim.py --data_path ./data/isic18/ --out_dir ./faint_sim_preview/
"""

import numpy as np
import cv2
import os
import argparse


def simulate_faint(img, mask_bin, fade=0.7, preserve_core=True, core_ratio=0.15):
    """Apply faint-lesion simulation to a single image."""
    img_f = img.astype(np.float32)

    # Skin color
    skin_mask = 1.0 - mask_bin
    skin_color = np.zeros(3)
    for c in range(3):
        skin_color[c] = (img_f[:, :, c] * skin_mask).sum() / (skin_mask.sum() + 1e-8)

    # Fade mask
    fade_mask = mask_bin.copy()
    if preserve_core:
        area = mask_bin.sum()
        erode_iter = max(1, int(np.sqrt(area * core_ratio) / 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        core = cv2.erode(mask_bin, kernel, iterations=erode_iter)
        fade_mask = np.clip(fade_mask - core, 0, 1)

    fade_mask = cv2.GaussianBlur(fade_mask, (15, 15), 0)
    fade_3d = fade_mask[:, :, np.newaxis]
    skin_img = np.ones_like(img_f) * skin_color

    result = img_f * (1 - fade * fade_3d) + skin_img * (fade * fade_3d)
    return np.clip(result, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/isic18/')
    parser.add_argument('--out_dir', default='./faint_sim_preview/')
    parser.add_argument('--n_samples', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    img_dir = os.path.join(args.data_path, 'train', 'images')
    mask_dir = os.path.join(args.data_path, 'train', 'masks')

    if not os.path.exists(img_dir):
        # Try .npy format
        img_npy = os.path.join(args.data_path, 'train_np', 'images')
        mask_npy = os.path.join(args.data_path, 'train_np', 'masks')
        if os.path.exists(img_npy):
            img_dir, mask_dir = img_npy, mask_npy

    files = sorted(os.listdir(img_dir))[:args.n_samples]

    for fname in files:
        img_path = os.path.join(img_dir, fname)

        if fname.endswith('.npy'):
            img_rgb = np.load(img_path)
            mask_path = os.path.join(mask_dir, fname)
            mask_raw = np.load(mask_path)
        else:
            img_rgb = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            mask_name = fname.replace('.jpg', '_segmentation.png').replace('.png', '_segmentation.png')
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, fname)
            mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask_raw is None or img_rgb is None:
            continue

        # Binarize mask
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[:, :, 0]
        mask_bin = (mask_raw > 127).astype(np.float32) if mask_raw.max() > 1 else \
                   (mask_raw > 0.5).astype(np.float32)

        if mask_bin.sum() < 100:
            continue

        # Generate faded versions
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        f50 = simulate_faint(img_bgr, mask_bin, fade=0.5, preserve_core=True)
        f70 = simulate_faint(img_bgr, mask_bin, fade=0.7, preserve_core=True)
        f85 = simulate_faint(img_bgr, mask_bin, fade=0.85, preserve_core=True)
        f85_no_core = simulate_faint(img_bgr, mask_bin, fade=0.85, preserve_core=False)

        # Mask overlay for reference
        mask_vis = cv2.cvtColor((mask_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        for label, panel in [("Original", img_bgr), ("GT mask", mask_vis),
                              ("Fade 50%", f50), ("Fade 70%", f70),
                              ("Fade 85% +core", f85), ("Fade 85% flat", f85_no_core)]:
            cv2.putText(panel, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        composite = np.hstack([img_bgr, mask_vis, f50, f70, f85, f85_no_core])
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(args.out_dir, f"faint_{base}.png")
        cv2.imwrite(out_path, composite)
        print(f"Saved: {out_path}")

    print(f"\nDone! Check {args.out_dir}/")
    print("\nWhat to look for:")
    print("  - 'Fade 85% +core': dark center remains, periphery fades to skin color")
    print("    This simulates the exact donut pattern you're struggling with")
    print("  - 'Fade 85% flat': entire lesion fades uniformly")
    print("    This simulates very faint flat lesions")
    print("  - The GT mask is UNCHANGED — model must still predict the full boundary")


if __name__ == '__main__':
    main()
