"""
Advanced faint-lesion simulator with natural random boundaries.

Combines the fading concept with shape diversity: instead of using the
smooth GT mask boundary (which looks artificial), we generate random
irregular fade regions with:
  - Perlin-noise-distorted boundaries for organic edges
  - True donut holes (faint outer ring, original/different inner)
  - Multi-focal patterns (several disconnected faint blobs)
  - Variable fade gradients (not uniform — patchy like real lesions)

Each training image can generate multiple augmented variants for
effective dataset expansion.
"""

import numpy as np
import cv2
import random
import math


class myFaintLesionAugmentor:
    """
    Generate realistic synthetic faint lesions during training.

    For each image:
    1. Read the GT mask to locate the lesion
    2. Generate a RANDOM fade region that roughly covers the lesion
       but with irregular, natural-looking boundaries
    3. Optionally punch donut holes or split into multi-focal blobs
    4. Fade the selected pixels toward the local skin color
    5. Return the faded image with the ORIGINAL mask unchanged

    Parameters
    ----------
    p : float
        Probability of applying augmentation per image.
    fade_range : tuple
        Min/max fade strength (0=no fade, 1=invisible).
    mode_probs : dict
        Probability of each pattern type:
        'diffuse' — single irregular blob, fully faded
        'donut'   — faded ring with restored/original center
        'multi'   — 2-4 disconnected faded blobs
        'partial' — only part of the lesion fades (asymmetric)
    boundary_roughness : float
        How jagged the fade boundary is (0=smooth, 1=very rough).
    n_variants : int
        Number of augmented copies to generate per image (for offline
        dataset expansion). Set to 1 for online augmentation.
    """

    def __init__(
        self,
        p: float = 0.5,
        fade_range: tuple = (0.4, 0.9),
        mode_probs: dict = None,
        boundary_roughness: float = 0.5,
    ):
        self.p = p
        self.fade_range = fade_range
        self.mode_probs = mode_probs or {
            'diffuse': 0.3,
            'donut': 0.3,
            'multi': 0.2,
            'partial': 0.2,
        }
        self.boundary_roughness = boundary_roughness

    def _get_mask_binary(self, mask):
        """Extract binary mask from whatever format the GT is in."""
        if mask.ndim == 3:
            m = mask[:, :, 0]
        else:
            m = mask
        if m.max() > 1:
            return (m > 127).astype(np.float32)
        return (m > 0.5).astype(np.float32)

    def _get_lesion_bbox(self, mask_bin):
        """Get bounding box of the lesion region."""
        ys, xs = np.where(mask_bin > 0.5)
        if len(ys) == 0:
            return None
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

    def _compute_skin_color(self, img, mask_bin):
        """Compute local skin color from non-lesion pixels."""
        skin = 1.0 - mask_bin
        if skin.sum() < 50:
            return np.array([180, 160, 155], dtype=np.float32)
        color = np.zeros(3, dtype=np.float32)
        for c in range(3):
            color[c] = (img[:, :, c] * skin).sum() / (skin.sum() + 1e-8)
        return color

    def _random_irregular_mask(self, h, w, center_y, center_x, radius_y, radius_x):
        """
        Generate a random irregular blob mask using deformed ellipse
        with polygon perturbation for natural-looking boundaries.
        """
        mask = np.zeros((h, w), dtype=np.float32)

        # Generate random polygon points around the center
        n_points = random.randint(8, 20)
        points = []
        for i in range(n_points):
            angle = 2 * math.pi * i / n_points
            # Random radial perturbation for irregular boundary
            r_factor = 1.0 + random.gauss(0, self.boundary_roughness * 0.4)
            r_factor = max(0.3, min(1.8, r_factor))
            px = center_x + radius_x * r_factor * math.cos(angle)
            py = center_y + radius_y * r_factor * math.sin(angle)
            points.append([int(px), int(py)])

        pts = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask, [hull], 1.0)

        # Blur for soft edges
        blur_k = max(3, int(min(radius_y, radius_x) * 0.3)) | 1
        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)

        return mask

    def _generate_diffuse(self, h, w, mask_bin, bbox):
        """Single irregular blob covering most of the lesion."""
        y1, x1, y2, x2 = bbox
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        # Slightly expand beyond lesion boundary for realistic overlap
        expand = random.uniform(1.0, 1.4)
        fade_mask = self._random_irregular_mask(
            h, w, cy, cx, int(ry * expand), int(rx * expand)
        )
        # Constrain to be mostly within the lesion area (but allow some bleed)
        bleed = random.uniform(0.0, 0.3)
        dilated = cv2.dilate(mask_bin, np.ones((7, 7)), iterations=2)
        constraint = mask_bin * (1 - bleed) + dilated * bleed
        constraint = cv2.GaussianBlur(constraint, (11, 11), 0)
        fade_mask = fade_mask * np.clip(constraint + 0.1, 0, 1)

        return fade_mask

    def _generate_donut(self, h, w, mask_bin, bbox):
        """Faded outer ring with a preserved (original) center — true donut."""
        y1, x1, y2, x2 = bbox
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        # Outer ring: irregular blob covering the lesion
        outer = self._generate_diffuse(h, w, mask_bin, bbox)

        # Inner hole: smaller irregular blob centered on the lesion
        core_scale = random.uniform(0.2, 0.5)
        inner = self._random_irregular_mask(
            h, w, cy + random.randint(-ry//4, ry//4),
            cx + random.randint(-rx//4, rx//4),
            max(5, int(ry * core_scale)),
            max(5, int(rx * core_scale))
        )

        # Donut = outer minus inner
        fade_mask = np.clip(outer - inner * random.uniform(0.7, 1.0), 0, 1)

        return fade_mask

    def _generate_multi(self, h, w, mask_bin, bbox):
        """Multiple disconnected faded blobs — multi-focal pattern."""
        y1, x1, y2, x2 = bbox
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        n_blobs = random.randint(2, 4)
        combined = np.zeros((h, w), dtype=np.float32)

        for _ in range(n_blobs):
            # Random center within or near the lesion bbox
            by = random.randint(max(0, y1 - ry//2), min(h-1, y2 + ry//2))
            bx = random.randint(max(0, x1 - rx//2), min(w-1, x2 + rx//2))
            # Random size (smaller than full lesion)
            scale = random.uniform(0.15, 0.5)
            blob = self._random_irregular_mask(
                h, w, by, bx,
                max(5, int(ry * scale)),
                max(5, int(rx * scale))
            )
            combined = np.maximum(combined, blob)

        # Intersect with dilated lesion region
        dilated = cv2.dilate(mask_bin, np.ones((7, 7)), iterations=3)
        dilated = cv2.GaussianBlur(dilated, (15, 15), 0)
        combined = combined * np.clip(dilated + 0.05, 0, 1)

        return combined

    def _generate_partial(self, h, w, mask_bin, bbox):
        """Only part of the lesion fades — asymmetric pattern."""
        y1, x1, y2, x2 = bbox
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        # Offset center to one side
        offset_y = random.randint(-ry//2, ry//2)
        offset_x = random.randint(-rx//2, rx//2)

        fade_mask = self._random_irregular_mask(
            h, w,
            cy + offset_y, cx + offset_x,
            max(5, int(ry * random.uniform(0.4, 0.8))),
            max(5, int(rx * random.uniform(0.4, 0.8)))
        )
        # Must overlap with lesion
        fade_mask = fade_mask * mask_bin

        return fade_mask

    def _add_patchy_texture(self, fade_mask, h, w):
        """Add patchiness to the fade for non-uniform fading (like real lesions)."""
        # Random low-frequency noise
        small = np.random.rand(h // 16 + 1, w // 16 + 1).astype(np.float32)
        noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = cv2.GaussianBlur(noise, (31, 31), 0)
        # Normalize to 0.5-1.0 range (modulate fade strength, don't zero it)
        noise = 0.5 + 0.5 * (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return fade_mask * noise

    def __call__(self, data):
        image, mask = data

        if random.random() > self.p:
            return image, mask

        mask_bin = self._get_mask_binary(mask)
        bbox = self._get_lesion_bbox(mask_bin)
        if bbox is None:
            return image, mask

        h, w = image.shape[:2]
        was_uint8 = image.dtype == np.uint8
        img = image.astype(np.float32)

        skin_color = self._compute_skin_color(img, mask_bin)

        # Choose pattern mode
        r = random.random()
        cumulative = 0
        mode = 'diffuse'
        for m, prob in self.mode_probs.items():
            cumulative += prob
            if r < cumulative:
                mode = m
                break

        # Generate fade mask based on mode
        if mode == 'donut':
            fade_mask = self._generate_donut(h, w, mask_bin, bbox)
        elif mode == 'multi':
            fade_mask = self._generate_multi(h, w, mask_bin, bbox)
        elif mode == 'partial':
            fade_mask = self._generate_partial(h, w, mask_bin, bbox)
        else:
            fade_mask = self._generate_diffuse(h, w, mask_bin, bbox)

        # Add patchy texture for realism
        if random.random() < 0.6:
            fade_mask = self._add_patchy_texture(fade_mask, h, w)

        # Apply fade
        fade_strength = random.uniform(*self.fade_range)
        fade_3d = (fade_mask * fade_strength)[:, :, np.newaxis]
        skin_img = np.ones_like(img) * skin_color[np.newaxis, np.newaxis, :]

        result = img * (1 - fade_3d) + skin_img * fade_3d

        if was_uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result, mask  # mask UNCHANGED


def generate_expanded_dataset(
    img_dir, mask_dir, out_img_dir, out_mask_dir,
    variants_per_image=3, fade_range=(0.3, 0.9)
):
    """
    Offline dataset expansion: generate multiple faint variants of each
    training image and save them as new training samples.

    This multiplies your effective faint-lesion training data.

    Parameters
    ----------
    img_dir, mask_dir : str
        Paths to original training images and masks (.npy or image files).
    out_img_dir, out_mask_dir : str
        Where to save generated variants.
    variants_per_image : int
        How many faint variants to generate per original image.
    """
    import os
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    augmentor = myFaintLesionAugmentor(
        p=1.0,  # always apply for offline generation
        fade_range=fade_range,
    )

    files = sorted(os.listdir(img_dir))
    total = 0

    for fname in files:
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if fname.endswith('.npy'):
            img = np.load(img_path)
            msk = np.load(mask_path)
        else:
            img = cv2.imread(img_path)
            msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or msk is None:
            continue

        for v in range(variants_per_image):
            aug_img, aug_msk = augmentor((img.copy(), msk.copy()))

            base = os.path.splitext(fname)[0]
            ext = os.path.splitext(fname)[1]
            out_name = f"{base}_faint_v{v}{ext}"

            if fname.endswith('.npy'):
                np.save(os.path.join(out_img_dir, out_name), aug_img)
                np.save(os.path.join(out_mask_dir, out_name), aug_msk)
            else:
                cv2.imwrite(os.path.join(out_img_dir, out_name), aug_img)
                cv2.imwrite(os.path.join(out_mask_dir, out_name), aug_msk)
            total += 1

    print(f"Generated {total} faint-lesion variants in {out_img_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preview or expand faint-lesion augmentation')
    parser.add_argument('--mode', choices=['preview', 'expand'], default='preview')
    parser.add_argument('--data_path', default='./data/isic18/')
    parser.add_argument('--out_dir', default='./faint_preview/')
    parser.add_argument('--variants', type=int, default=3)
    parser.add_argument('--n_samples', type=int, default=8)
    args = parser.parse_args()

    import os

    img_dir = os.path.join(args.data_path, 'train_np', 'images')
    mask_dir = os.path.join(args.data_path, 'train_np', 'masks')

    if not os.path.exists(img_dir):
        img_dir = os.path.join(args.data_path, 'train', 'images')
        mask_dir = os.path.join(args.data_path, 'train', 'masks')

    if args.mode == 'preview':
        os.makedirs(args.out_dir, exist_ok=True)
        augmentor = myFaintLesionAugmentor(p=1.0, fade_range=(0.4, 0.9))

        files = sorted(os.listdir(img_dir))[:args.n_samples]
        for fname in files:
            ip = os.path.join(img_dir, fname)
            mp = os.path.join(mask_dir, fname)

            if fname.endswith('.npy'):
                img = np.load(ip)
                msk = np.load(mp)
            else:
                img = cv2.imread(ip)
                msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

            if img is None or msk is None:
                continue

            # Generate 6 variants to show diversity
            panels = [img.copy()]
            labels = ['Original']
            for i in range(6):
                aug, _ = augmentor((img.copy(), msk.copy()))
                panels.append(aug)
                labels.append(f'Variant {i+1}')

            # Add mask visualization
            if msk.ndim == 2:
                mask_vis = cv2.cvtColor(
                    ((msk > (127 if msk.max() > 1 else 0.5)).astype(np.uint8) * 255),
                    cv2.COLOR_GRAY2BGR if img.ndim == 3 else cv2.COLOR_GRAY2GRAY
                )
                if mask_vis.ndim == 2:
                    mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
            else:
                mask_vis = msk.copy()

            for label, panel in zip(labels, panels):
                if panel.ndim == 3 and panel.shape[2] == 3:
                    cv2.putText(panel, label, (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            row1 = np.hstack(panels[:4])
            row2_panels = panels[4:] + [mask_vis]
            # Pad if needed
            while len(row2_panels) < 4:
                row2_panels.append(np.zeros_like(panels[0]))
            row2 = np.hstack(row2_panels[:4])
            composite = np.vstack([row1, row2])

            base = os.path.splitext(fname)[0]
            out = os.path.join(args.out_dir, f"preview_{base}.png")
            if composite.ndim == 3:
                cv2.imwrite(out, composite if composite.shape[2] == 3 else
                           cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(out, composite)
            print(f"Saved: {out}")

        print(f"\nPreview done! Check {args.out_dir}/")
        print("Each image shows 6 random variants with different patterns")
        print("(diffuse, donut, multi-focal, partial)")

    elif args.mode == 'expand':
        out_img = os.path.join(args.out_dir, 'images')
        out_msk = os.path.join(args.out_dir, 'masks')
        generate_expanded_dataset(
            img_dir, mask_dir, out_img, out_msk,
            variants_per_image=args.variants,
        )
        print(f"\nExpanded dataset saved to {args.out_dir}/")
        print("Copy these into your training data directory to expand the dataset")
