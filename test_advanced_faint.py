import numpy as np
import cv2
import os
import argparse
from PIL import Image

# 假设你已经把 myFaintLesionAugmentor 类放进了 utils.py
# 如果还没放，可以直接把类定义贴在这个脚本开头
"""
myFaintLesionAugmentor — drop-in pipeline transform.

Five pattern modes:
  diffuse   — single irregular faded blob
  donut     — faded ring with random-shaped preserved center
  multi     — 2-4 disconnected faded blobs (multi-focal)
  partial   — asymmetric fade on one side only
  shattered — dense overlapping holes that fragment the lesion

Usage in config_setting.py train_transformer:
    myFaintLesionAugmentor(p=0.5, fade_range=(0.4, 0.9))

Insert AFTER myAdvancedSkinCutout, BEFORE myNormalize.
"""

import numpy as np
import cv2
import random
import math


class myFaintLesionAugmentor:

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
            'diffuse':   0.20,
            'donut':     0.25,
            'multi':     0.20,
            'partial':   0.10,
            'shattered': 0.25,
        }
        self.boundary_roughness = boundary_roughness

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mask_bin(self, mask):
        m = mask[:, :, 0] if mask.ndim == 3 else mask
        return (m > (127 if m.max() > 1 else 0.5)).astype(np.float32)

    def _bbox(self, m):
        ys, xs = np.where(m > 0.5)
        if len(ys) == 0:
            return None
        return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())

    def _skin_color(self, img, m):
        skin = 1.0 - m
        if skin.sum() < 50:
            return img.mean(axis=(0, 1))
        c = np.zeros(3, dtype=np.float32)
        for i in range(3):
            c[i] = (img[:, :, i] * skin).sum() / (skin.sum() + 1e-8)
        return c

    def _irregular_blob(self, h, w, cy, cx, ry, rx):
        """Random convex blob with jagged boundary."""
        mask = np.zeros((h, w), dtype=np.float32)
        n = random.randint(8, 18)
        pts = []
        for i in range(n):
            a = 2 * math.pi * i / n
            rf = max(0.3, min(1.8, 1.0 + random.gauss(0, self.boundary_roughness * 0.4)))
            pts.append([
                int(np.clip(cx + rx * rf * math.cos(a), 0, w - 1)),
                int(np.clip(cy + ry * rf * math.sin(a), 0, h - 1)),
            ])
        hull = cv2.convexHull(np.array(pts, dtype=np.int32))
        cv2.fillPoly(mask, [hull], 1.0)
        k = max(3, int(min(ry, rx) * 0.25)) | 1
        return cv2.GaussianBlur(mask, (k, k), 0)

    def _random_shape(self, h, w, cy, cx, ry, rx):
        """Random shape: ellipse, polygon, or blob — for variety."""
        choice = random.choice(['blob', 'ellipse', 'poly'])
        mask = np.zeros((h, w), dtype=np.uint8)

        if choice == 'ellipse':
            angle = random.uniform(0, 360)
            cv2.ellipse(mask, (int(cx), int(cy)),
                        (max(1, int(rx)), max(1, int(ry))),
                        angle, 0, 360, 1, -1)
            k = max(3, int(min(ry, rx) * 0.2)) | 1
            return cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)

        elif choice == 'poly':
            n = random.randint(3, 7)
            pts = np.array([
                [random.randint(max(0, int(cx - rx)), min(w - 1, int(cx + rx))),
                 random.randint(max(0, int(cy - ry)), min(h - 1, int(cy + ry)))]
                for _ in range(n)
            ], dtype=np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillPoly(mask, [hull], 1)
            k = max(3, int(min(ry, rx) * 0.2)) | 1
            return cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)

        else:
            return self._irregular_blob(h, w, cy, cx, ry, rx)

    def _patchy(self, fade_mask, h, w):
        """Non-uniform fade intensity for realism."""
        small = np.random.rand(h // 16 + 1, w // 16 + 1).astype(np.float32)
        noise = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = cv2.GaussianBlur(noise, (31, 31), 0)
        noise = 0.5 + 0.5 * (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return fade_mask * noise

    # ------------------------------------------------------------------
    # Pattern generators
    # ------------------------------------------------------------------
    def _gen_diffuse(self, h, w, m, bb):
        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        expand = random.uniform(1.0, 1.3)
        blob = self._irregular_blob(h, w, cy, cx,
                                     int((y2 - y1) / 2 * expand),
                                     int((x2 - x1) / 2 * expand))
        dilated = cv2.dilate(m, np.ones((5, 5)), iterations=2)
        dilated = cv2.GaussianBlur(dilated, (11, 11), 0)
        return blob * np.clip(dilated + 0.1, 0, 1)

    def _gen_donut(self, h, w, m, bb):
        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        outer = self._gen_diffuse(h, w, m, bb)

        core_s = random.uniform(0.15, 0.45)
        oy = random.randint(-max(1, ry // 4), max(1, ry // 4))
        ox = random.randint(-max(1, rx // 4), max(1, rx // 4))
        inner = self._random_shape(h, w, cy + oy, cx + ox,
                                    max(5, ry * core_s), max(5, rx * core_s))
        return np.clip(outer - inner * random.uniform(0.7, 1.0), 0, 1)

    def _gen_multi(self, h, w, m, bb):
        y1, x1, y2, x2 = bb
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
        combined = np.zeros((h, w), dtype=np.float32)

        for _ in range(random.randint(2, 4)):
            by = random.randint(max(0, y1 - ry // 3), min(h - 1, y2 + ry // 3))
            bx = random.randint(max(0, x1 - rx // 3), min(w - 1, x2 + rx // 3))
            s = random.uniform(0.15, 0.45)
            blob = self._random_shape(h, w, by, bx, max(5, ry * s), max(5, rx * s))
            combined = np.maximum(combined, blob)

        dilated = cv2.dilate(m, np.ones((5, 5)), iterations=3)
        dilated = cv2.GaussianBlur(dilated, (15, 15), 0)
        return combined * np.clip(dilated + 0.05, 0, 1)

    def _gen_partial(self, h, w, m, bb):
        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        oy = random.randint(-ry // 2, ry // 2)
        ox = random.randint(-rx // 2, rx // 2)
        blob = self._random_shape(h, w, cy + oy, cx + ox,
                                   max(5, int(ry * random.uniform(0.3, 0.7))),
                                   max(5, int(rx * random.uniform(0.3, 0.7))))
        return blob * m

    def _gen_shattered(self, h, w, m, bb):
        """
        Dense overlapping holes that fragment the lesion.

        Creates 5-12 small random holes scattered across the lesion,
        each one "restoring" (fading) a patch to skin color. The result
        looks like the lesion has been broken into irregular shards
        with skin-colored gaps between them.
        """
        y1, x1, y2, x2 = bb
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        n_holes = random.randint(5, 12)
        combined = np.zeros((h, w), dtype=np.float32)

        for _ in range(n_holes):
            # Random position within the lesion bbox
            hy = random.randint(y1, y2)
            hx = random.randint(x1, x2)
            # Small random size (5-25% of lesion dimension)
            scale = random.uniform(0.05, 0.25)
            hole = self._random_shape(
                h, w, hy, hx,
                max(3, int(ry * scale)),
                max(3, int(rx * scale))
            )
            # Random intensity per hole for variety
            combined = np.maximum(combined, hole * random.uniform(0.6, 1.0))

        # Strictly within lesion area (no bleed outside)
        blurred_mask = cv2.GaussianBlur(m, (7, 7), 0)
        return combined * blurred_mask

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------
    def __call__(self, data):
        image, mask = data

        if random.random() > self.p:
            return image, mask

        m = self._mask_bin(mask)
        bb = self._bbox(m)
        if bb is None or m.sum() < 100:
            return image, mask

        h, w = image.shape[:2]
        was_uint8 = image.dtype == np.uint8
        img = image.astype(np.float32)
        skin = self._skin_color(img, m)

        # Pick mode
        r = random.random()
        cum = 0
        mode = 'diffuse'
        for name, prob in self.mode_probs.items():
            cum += prob
            if r < cum:
                mode = name
                break

        # Generate pattern
        gen = {
            'diffuse':   self._gen_diffuse,
            'donut':     self._gen_donut,
            'multi':     self._gen_multi,
            'partial':   self._gen_partial,
            'shattered': self._gen_shattered,
        }
        fade_mask = gen[mode](h, w, m, bb)

        # Patchy texture (60% of the time)
        if random.random() < 0.6:
            fade_mask = self._patchy(fade_mask, h, w)

        # Fade toward skin color
        strength = random.uniform(*self.fade_range)
        fade_3d = (fade_mask * strength)[:, :, np.newaxis]
        skin_img = np.ones_like(img) * skin[np.newaxis, np.newaxis, :]
        result = img * (1 - fade_3d) + skin_img * fade_3d

        if was_uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)

        return result, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/isic18/')
    parser.add_argument('--out_dir', default='./advanced_faint_preview/')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 初始化增强器，设置 p=1.0 确保每次都触发，方便我们观察
    augmentor = myFaintLesionAugmentor(p=1.0, fade_range=(0.6, 0.9))

    # 设置路径（兼容你之前的结构）
    img_dir = os.path.join(args.data_path, 'train', 'images')
    mask_dir = os.path.join(args.data_path, 'train', 'masks')

    files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]

    for fname in files:
        # 读取原图和掩码
        img = cv2.imread(os.path.join(img_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_name = fname.replace('.jpg', '_segmentation.png').replace('.png', '_segmentation.png')
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(mask_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 准备对比列表
        # 我们手动指定模式来测试
        modes = ['diffuse', 'donut', 'multi', 'partial', 'shattered']
        results = [img]  # 第一个是原图
        labels = ["Original"]

        for mode in modes:
            # 临时修改增强器的模式概率，强行指定某种模式
            augmentor.mode_probs = {m: (1.0 if m == mode else 0.0) for m in modes}
            aug_img, _ = augmentor((img.copy(), mask.copy()))
            results.append(aug_img)
            labels.append(mode.capitalize())

        # 拼接结果
        vis_results = []
        for i, res in enumerate(results):
            res_bgr = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            cv2.putText(res_bgr, labels[i], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            vis_results.append(res_bgr)

        composite = np.hstack(vis_results)
        save_path = os.path.join(args.out_dir, f"test_{fname}")
        cv2.imwrite(save_path, composite)
        print(f"Saved visualization to: {save_path}")

    print(f"\n预览完成！请查看文件夹: {args.out_dir}")
    print("重点观察 'Shattered' 面板，看那些洞洞的随机感和破碎感是否达到了你的要求。")


if __name__ == '__main__':
    main()