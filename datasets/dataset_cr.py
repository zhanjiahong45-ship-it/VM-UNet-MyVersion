"""
NPY_datasets_CR
================
为 Consistency Reinforcement (CR) 训练设计的 Dataset 类。

每次 __getitem__ 返回一个三元组：
    (img_normal, img_aug, msk)

img_normal 与 img_aug 共用同一组随机翻转/旋转参数（保证 GT 对齐），
唯一区别是 img_aug 多走了一步 myFaintLesionAugmentor 的"病灶定向褪色"。

Usage in trainff.py:
    from datasets.dataset_cr import NPY_datasets_CR
    train_dataset = NPY_datasets_CR(config.data_path, config, train=True)
    # val/test 仍然用原 NPY_datasets

Note: 该 dataset 专用于 train, val/test 应继续用原 NPY_datasets
"""

import os
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from utils import myFaintLesionAugmentor, myNormalize, myToTensor, myResize


class NPY_datasets_CR(Dataset):
    """
    CR 版 Dataset：每个样本返回 (img_normal, img_aug, msk).

    设计要点:
    - 翻转/旋转/resize 在 numpy/PIL 阶段做，对 normal 和 aug 用同一组随机参数
    - 褪色仅作用在 aug 分支，且在 normalize 之前（uint8 [0,255] 域）
    - normalize / ToTensor 各自分别走完
    - val/test 不应使用此类，请仍用原 NPY_datasets
    """

    def __init__(self, path_Data, config, train=True,
                 fade_p=0.75, fade_range=(0.3, 0.95), mode_probs=None):
        super().__init__()
        assert train, "NPY_datasets_CR is only for training. Use NPY_datasets for val/test."

        # 收集训练样本路径
        images_dir = os.path.join(path_Data, 'train', 'images')
        masks_dir = os.path.join(path_Data, 'train', 'masks')
        images_list = sorted(os.listdir(images_dir))
        masks_list = sorted(os.listdir(masks_dir))
        self.data = []
        for i in range(len(images_list)):
            self.data.append([
                os.path.join(images_dir, images_list[i]),
                os.path.join(masks_dir, masks_list[i]),
            ])

        # 病灶定向褪色器（训练时用）
        self.fade_aug = myFaintLesionAugmentor(
            p=fade_p,
            fade_range=fade_range,
            mode_probs=mode_probs,
        )

        # 数据归一化和尺寸（与原 train_transformer 保持一致）
        self.normalize = myNormalize(config.datasets, train=True)
        self.to_tensor = myToTensor()
        self.resize = myResize(config.input_size_h, config.input_size_w)

        # 翻转/旋转概率（与原 config 一致）
        self.hflip_p = 0.5
        self.vflip_p = 0.5
        self.rotate_p = 0.5
        self.rotate_range = (0.0, 360.0)

    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------------------
    # 内部工具：对一对 (img, msk) 顺序应用 normalize → ToTensor → resize
    # ------------------------------------------------------------------
    def _finalize(self, img, msk):
        img, msk = self.normalize((img, msk))
        img, msk = self.to_tensor((img, msk))
        img, msk = self.resize((img, msk))
        return img, msk

    def __getitem__(self, idx):
        img_path, msk_path = self.data[idx]
        img = np.array(Image.open(img_path).convert('RGB'))           # uint8 [0,255]
        msk = np.expand_dims(
            np.array(Image.open(msk_path).convert('L')), axis=2
        ) / 255.0                                                       # float [0,1]

        # ----------------------------------------------------------
        # 1) 同一组随机几何变换：先在 PIL/numpy 域做，保证两路一致
        # ----------------------------------------------------------
        # ---- 水平翻转 ----
        if random.random() < self.hflip_p:
            img = np.ascontiguousarray(img[:, ::-1, :])
            msk = np.ascontiguousarray(msk[:, ::-1, :])

        # ---- 垂直翻转 ----
        if random.random() < self.vflip_p:
            img = np.ascontiguousarray(img[::-1, :, :])
            msk = np.ascontiguousarray(msk[::-1, :, :])

        # ---- 随机旋转 ----
        if random.random() < self.rotate_p:
            angle = random.uniform(*self.rotate_range)
            # 用 PIL 旋转保持图像质量；mask 用 nearest 避免插值出非 0/1 值
            img_pil = Image.fromarray(img)
            img_pil = img_pil.rotate(angle, resample=Image.BILINEAR)
            img = np.array(img_pil)

            msk_pil = Image.fromarray((msk[:, :, 0] * 255).astype(np.uint8))
            msk_pil = msk_pil.rotate(angle, resample=Image.NEAREST)
            msk = np.expand_dims(np.array(msk_pil), axis=2) / 255.0

        # ----------------------------------------------------------
        # 2) 此时 img 和 msk 是经过同样几何变换后的版本
        #    复制一份给"褪色分支"
        # ----------------------------------------------------------
        img_normal = img.copy()
        msk_normal = msk.copy()

        # 褪色仅作用在 img_aug 上, msk_aug 保持与 msk_normal 相同（GT 不变）
        img_aug, msk_aug = self.fade_aug(
            (img.copy(), msk.copy())  # fade_aug 内部根据 self.p 决定是否褪色
        )

        # ----------------------------------------------------------
        # 3) 各自走完 normalize → ToTensor → resize
        # ----------------------------------------------------------
        img_normal_t, msk_t = self._finalize(img_normal, msk_normal)
        img_aug_t, _ = self._finalize(img_aug, msk_aug)

        return img_normal_t, img_aug_t, msk_t