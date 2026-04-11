import numpy as np
import cv2
import os
import argparse
"""
myFaintLesionAugmentor — production version.

Drop into utils.py. One line in config_setting.py:

    myFaintLesionAugmentor(p=0.75)

p=0.75 gives the 3:1 faded-to-original ratio you want. Each call
produces one unique random variant. Over 120 epochs → ~90 unique
faint patterns per image, all different. Better diversity than any
fixed offline expansion.

Six pattern modes
-----------------
diffuse    — whole lesion fades toward skin, patchy/uneven
donut      — faded ring + preserved random center
nested     — recursive donut-in-donut (2-3 layers)
multi      — 2-4 disconnected faded blobs
shattered  — 5-15 overlapping holes fragmenting the lesion
ghost      — COMPLETE fadeout, lesion becomes nearly invisible

All parameters are fully randomized per call: shape type, size,
position, count, fade strength, boundary roughness, patchiness.
"""

import numpy as np
import cv2
import random
import math


class myFaintLesionAugmentor:

    def __init__(
        self,
        p: float = 0.75,
        fade_range: tuple = (0.3, 0.95),
        mode_probs: dict = None,
    ):
        self.p = p
        self.fade_range = fade_range
        self.mode_probs = mode_probs or {
            'diffuse':   0.15,
            'donut':     0.15,
            'nested':    0.15,
            'multi':     0.15,
            'shattered': 0.25,
            'ghost':     0.15,
        }

    # ==================================================================
    # Shape primitives — fully randomized
    # ==================================================================
    def _rand_shape(self, h, w, cy, cx, ry, rx):
        """One random filled shape. Type, vertex count, rotation all random."""
        mask = np.zeros((h, w), dtype=np.float32)
        kind = random.choice(['ellipse', 'poly', 'blob'])

        # Clamp center and radii to valid ranges
        ry, rx = max(3, int(ry)), max(3, int(rx))
        cy = int(np.clip(cy, ry, h - ry - 1))
        cx = int(np.clip(cx, rx, w - rx - 1))

        if kind == 'ellipse':
            angle = random.uniform(0, 360)
            cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)

        elif kind == 'poly':
            n = random.randint(3, 8)
            pts = []
            for _ in range(n):
                px = random.randint(max(0, cx - rx), min(w - 1, cx + rx))
                py = random.randint(max(0, cy - ry), min(h - 1, cy + ry))
                pts.append([px, py])
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            cv2.fillPoly(mask, [hull], 1.0)

        else:  # blob — deformed ellipse with jagged boundary
            n = random.randint(10, 20)
            roughness = random.uniform(0.2, 0.6)
            pts = []
            for i in range(n):
                a = 2 * math.pi * i / n
                rf = max(0.3, 1.0 + random.gauss(0, roughness))
                px = int(np.clip(cx + rx * rf * math.cos(a), 0, w - 1))
                py = int(np.clip(cy + ry * rf * math.sin(a), 0, h - 1))
                pts.append([px, py])
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            cv2.fillPoly(mask, [hull], 1.0)

        # Random edge softness
        blur = max(3, random.choice([3, 5, 7, 9, 11, 15])) | 1
        return cv2.GaussianBlur(mask, (blur, blur), 0)

    def _rand_pos_in_bbox(self, bb, margin=0.3):
        """Random center within or near the lesion bbox."""
        y1, x1, y2, x2 = bb
        ry, rx = (y2 - y1) / 2, (x2 - x1) / 2
        cy = random.randint(int(y1 - ry * margin), int(y2 + ry * margin))
        cx = random.randint(int(x1 - rx * margin), int(x2 + rx * margin))
        return cy, cx

    def _rand_size(self, ry, rx, lo=0.1, hi=0.6):
        """Random size as fraction of lesion dimensions."""
        s = random.uniform(lo, hi)
        return max(4, int(ry * s)), max(4, int(rx * s))

    # ==================================================================
    # Texture: patchy non-uniform fade
    # ==================================================================
    def _patchy(self, mask, h, w):
        """Modulate fade mask with low-frequency noise for uneven fading."""
        scale = random.choice([8, 12, 16, 24])
        noise = np.random.rand(h // scale + 1, w // scale + 1).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        k = random.choice([15, 21, 31]) | 1
        noise = cv2.GaussianBlur(noise, (k, k), 0)
        lo = random.uniform(0.3, 0.6)
        noise = lo + (1 - lo) * (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return mask * noise

    # ==================================================================
    # Lesion constraint: keep fade within/near the lesion
    # ==================================================================
    def _constrain(self, fade, mask_bin, bleed=0.1):
        """Limit fade region to lesion area with optional slight bleed."""
        dilate_k = random.randint(1, 4)
        dilated = cv2.dilate(mask_bin, np.ones((5, 5)), iterations=dilate_k)
        dilated = cv2.GaussianBlur(dilated, (11, 11), 0)
        boundary = mask_bin * (1 - bleed) + dilated * bleed
        return fade * np.clip(boundary + 0.05, 0, 1)

    # ==================================================================
    # Six pattern generators
    # ==================================================================
    def _gen_diffuse(self, h, w, m, bb):
        """Whole lesion fades, patchy and uneven."""
        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
        expand = random.uniform(0.9, 1.4)
        blob = self._rand_shape(h, w, cy, cx,
                                 int(ry * expand), int(rx * expand))
        return self._constrain(blob, m, bleed=random.uniform(0, 0.2))

    def _gen_donut(self, h, w, m, bb):
        """Faded ring, random-shaped hole in the center."""
        outer = self._gen_diffuse(h, w, m, bb)

        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        iy, ix = self._rand_size(ry, rx, 0.15, 0.5)
        oy = random.randint(-max(1, ry // 3), max(1, ry // 3))
        ox = random.randint(-max(1, rx // 3), max(1, rx // 3))
        inner = self._rand_shape(h, w, cy + oy, cx + ox, iy, ix)

        return np.clip(outer - inner * random.uniform(0.6, 1.0), 0, 1)

    def _gen_nested(self, h, w, m, bb):
        """Recursive donut-in-donut: 2-3 concentric layers."""
        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        n_layers = random.randint(2, 3)
        result = np.zeros((h, w), dtype=np.float32)
        cur_ry, cur_rx = ry, rx
        fade_on = True  # alternates: fade, restore, fade, restore...

        for layer in range(n_layers):
            scale = random.uniform(0.5, 0.8) if layer > 0 else random.uniform(0.9, 1.3)
            cur_ry = max(5, int(cur_ry * scale))
            cur_rx = max(5, int(cur_rx * scale))

            if cur_ry < 5 or cur_rx < 5:
                break

            oy = random.randint(-max(1, cur_ry // 4), max(1, cur_ry // 4))
            ox = random.randint(-max(1, cur_rx // 4), max(1, cur_rx // 4))
            ring = self._rand_shape(h, w, cy + oy, cx + ox, cur_ry, cur_rx)

            if fade_on:
                result = np.maximum(result, ring)
            else:
                result = np.clip(result - ring * random.uniform(0.5, 1.0), 0, 1)

            fade_on = not fade_on

        return self._constrain(result, m, bleed=0.1)

    def _gen_multi(self, h, w, m, bb):
        """2-4 disconnected faded blobs."""
        y1, x1, y2, x2 = bb
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
        combined = np.zeros((h, w), dtype=np.float32)

        for _ in range(random.randint(2, 4)):
            cy, cx = self._rand_pos_in_bbox(bb, margin=0.2)
            sy, sx = self._rand_size(ry, rx, 0.15, 0.5)
            blob = self._rand_shape(h, w, cy, cx, sy, sx)
            combined = np.maximum(combined, blob * random.uniform(0.5, 1.0))

        return self._constrain(combined, m, bleed=0.15)

    def _gen_shattered(self, h, w, m, bb):
        """5-15 dense overlapping holes fragmenting the lesion."""
        y1, x1, y2, x2 = bb
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
        combined = np.zeros((h, w), dtype=np.float32)

        n_holes = random.randint(5, 15)
        for _ in range(n_holes):
            hy = random.randint(y1, y2)
            hx = random.randint(x1, x2)
            sy, sx = self._rand_size(ry, rx, 0.04, 0.25)
            hole = self._rand_shape(h, w, hy, hx, sy, sx)
            combined = np.maximum(combined, hole * random.uniform(0.4, 1.0))

        # Strictly within lesion
        blurred = cv2.GaussianBlur(m, (7, 7), 0)
        return combined * blurred

    def _gen_ghost(self, h, w, m, bb):
        """
        Complete fadeout — entire lesion becomes nearly invisible.
        This is the hardest mode: simulates the worst-case faint lesion
        where NOTHING is dark. The model must learn from pure texture.
        Forces fade_range to (0.85, 0.98) regardless of global setting.
        """
        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        expand = random.uniform(1.0, 1.3)
        blob = self._rand_shape(h, w, cy, cx,
                                 int(ry * expand), int(rx * expand))
        fade = self._constrain(blob, m, bleed=0.05)

        # Override to extreme fade
        self._ghost_override = True
        return fade

    # ==================================================================
    # Main
    # ==================================================================
    def _get_mask_bin(self, mask):
        m = mask[:, :, 0] if mask.ndim == 3 else mask
        return (m > (127 if m.max() > 1 else 0.5)).astype(np.float32)

    def __call__(self, data):
        image, mask = data

        if random.random() > self.p:
            return image, mask

        m = self._get_mask_bin(mask)
        ys, xs = np.where(m > 0.5)
        if len(ys) < 100:
            return image, mask
        bb = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))

        h, w = image.shape[:2]
        was_uint8 = image.dtype == np.uint8
        img = image.astype(np.float32)

        # Skin color from non-lesion pixels
        skin_mask = 1.0 - m
        skin_color = np.zeros(3, dtype=np.float32)
        if skin_mask.sum() > 50:
            for c in range(3):
                skin_color[c] = (img[:, :, c] * skin_mask).sum() / (skin_mask.sum() + 1e-8)
        else:
            skin_color[:] = img.mean(axis=(0, 1))

        # Pick random mode
        r = random.random()
        cum = 0
        mode = 'diffuse'
        for name, prob in self.mode_probs.items():
            cum += prob
            if r < cum:
                mode = name
                break

        # Generate
        self._ghost_override = False
        generators = {
            'diffuse':   self._gen_diffuse,
            'donut':     self._gen_donut,
            'nested':    self._gen_nested,
            'multi':     self._gen_multi,
            'shattered': self._gen_shattered,
            'ghost':     self._gen_ghost,
        }
        fade_mask = generators[mode](h, w, m, bb)

        # Patchy texture (most of the time for realism)
        if random.random() < 0.7:
            fade_mask = self._patchy(fade_mask, h, w)

        # Fade strength
        if self._ghost_override:
            strength = random.uniform(0.85, 0.98)
        else:
            strength = random.uniform(*self.fade_range)

        # Apply: blend lesion toward skin color
        fade_3d = (fade_mask * strength)[:, :, np.newaxis]
        skin_img = np.ones_like(img) * skin_color[np.newaxis, np.newaxis, :]
        result = img * (1 - fade_3d) + skin_img * fade_3d

        if was_uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result, mask  # mask UNCHANGED — model must predict full boundary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/isic18/')
    parser.add_argument('--out_dir', default='./prod_augmentor_test/')
    parser.add_argument('--n_samples', type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 初始化增强器：p=1.0 确保测试时 100% 触发
    augmentor = myFaintLesionAugmentor(p=1.0, fade_range=(0.6, 0.95))

    # 路径匹配你的 ISIC18 结构
    img_dir = os.path.join(args.data_path, 'train', 'images')
    mask_dir = os.path.join(args.data_path, 'train', 'masks')

    if not os.path.exists(img_dir):
        print(f"错误: 找不到路径 {img_dir}，请检查 --data_path 参数")
        return

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))][:args.n_samples]

    for fname in files:
        # 1. 读取原图和掩码
        img = cv2.imread(os.path.join(img_dir, fname))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_name = fname.replace('.jpg', '_segmentation.png').replace('.png', '_segmentation.png')
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None: continue

        # 2. 循环测试 6 种模式
        modes = ['diffuse', 'donut', 'nested', 'multi', 'shattered', 'ghost']
        panels = []

        # 添加原图作为对比
        orig_vis = img.copy()
        cv2.putText(orig_vis, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        panels.append(orig_vis)

        for mode in modes:
            # 强制指定当前模式进行测试
            augmentor.mode_probs = {m: (1.0 if m == mode else 0.0) for m in modes}

            # 运行增强
            aug_img_rgb, _ = augmentor((img_rgb.copy(), mask.copy()))

            # 转回 BGR 用于 OpenCV 保存
            aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(aug_img_bgr, mode.capitalize(), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            panels.append(aug_img_bgr)

        # 3. 横向拼接并保存
        composite = np.hstack(panels)
        save_path = os.path.join(args.out_dir, f"prod_test_{fname}")
        cv2.imwrite(save_path, composite)
        print(f"已保存对比图: {save_path}")

    print(f"\n测试完成！结果在: {args.out_dir}")
    print("重点检查:")
    print(" - Ghost: 病灶是否几乎看不见 (模拟 Image 1)")
    print(" - Shattered: 是否有你想要的'多个洞洞覆盖'的破碎感 (模拟 Image 2)")
    print(" - Nested: 观察'洞中洞'的分层淡化效果")


if __name__ == '__main__':
    main()