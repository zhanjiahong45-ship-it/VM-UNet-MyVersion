"""
Quick visual: see what CLAHE does to your faint lesion images.
Run this BEFORE retraining to verify the enhancement works.

Usage:
    python test_clahe.py --img_dir ./inputs/ --out_dir ./clahe_preview/
"""

import cv2
import numpy as np
import os
import argparse


def apply_clahe(img_bgr, clip_limit=3.0, tile_size=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='./inputs/')
    parser.add_argument('--out_dir', default='./clahe_preview/')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.img_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(os.path.join(args.img_dir, fname))

        # Multiple clip limits to compare
        c2 = apply_clahe(img, clip_limit=2.0)
        c3 = apply_clahe(img, clip_limit=3.0)
        c5 = apply_clahe(img, clip_limit=5.0)

        # Add labels
        h, w = img.shape[:2]
        for label, panel in [("Original", img), ("CLAHE 2.0", c2),
                              ("CLAHE 3.0", c3), ("CLAHE 5.0", c5)]:
            cv2.putText(panel, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        composite = np.hstack([img, c2, c3, c5])
        out_path = os.path.join(args.out_dir, f"clahe_{fname}")
        cv2.imwrite(out_path, composite)
        print(f"Saved: {out_path}")

    print(f"\nDone! Compare the panels in {args.out_dir}/")
    print("Pick the clip_limit where faint lesions become clearly visible")
    print("but artifacts are minimal. Usually 3.0 is the sweet spot.")


if __name__ == '__main__':
    main()