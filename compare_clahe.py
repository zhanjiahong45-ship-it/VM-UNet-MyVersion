"""
Compare: Original vs Full CLAHE vs AB-only CLAHE vs Blend-back
This shows why AB-channel targeting reduces noise while keeping lesion boost.

Usage:
    python compare_clahe.py --img_dir ./inputs/ --out_dir ./clahe_compare/
"""

import cv2
import numpy as np
import os
import argparse


def clahe_L(img_bgr, clip=3.0):
    """Standard CLAHE on L channel — amplifies everything."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 0] = c.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_AB(img_bgr, clip=3.0):
    """CLAHE on A+B channels only — targets melanin, ignores texture."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    lab[:, :, 1] = c.apply(lab[:, :, 1])
    lab[:, :, 2] = c.apply(lab[:, :, 2])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_blend(img_bgr, clip=4.0, blend=0.5):
    """CLAHE on L, blended 50/50 with original."""
    enhanced = clahe_L(img_bgr, clip)
    return cv2.addWeighted(img_bgr, 1 - blend, enhanced, blend, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='./inputs/')
    parser.add_argument('--out_dir', default='./clahe_compare/')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.img_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(os.path.join(args.img_dir, fname))
        panels = [
            ("Original", img),
            ("CLAHE-L 3.0 (noisy)", clahe_L(img, 3.0)),
            ("CLAHE-AB 3.0 (targeted)", clahe_AB(img, 3.0)),
            ("Blend-back 50%", clahe_blend(img, 4.0, 0.5)),
        ]

        for label, panel in panels:
            cv2.putText(panel, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        composite = np.hstack([p for _, p in panels])
        out = os.path.join(args.out_dir, f"compare_{fname}")
        cv2.imwrite(out, composite)
        print(f"Saved: {out}")

    print(f"\nDone! Compare in {args.out_dir}/")
    print("Look at the faint lesion area in each panel:")
    print("  - CLAHE-L: lesion visible but skin texture very noisy")
    print("  - CLAHE-AB: lesion visible, texture stays clean")
    print("  - Blend-back: middle ground")


if __name__ == '__main__':
    main()
