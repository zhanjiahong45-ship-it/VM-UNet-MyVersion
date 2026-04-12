from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom

# 👑 导入你编写的模糊病灶增强器
from utils import myFaintLesionAugmentor


class NPY_datasets(Dataset):
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self).__init__()
        self.train = train
        self.phase = 1  # 默认初始化为 Phase 1
        # 👑 Phase 1: 训练集设为 7N（1原图+6变体），验证集永远保持 1N
        self.expand_ratio = 7 if train else 1

        if train:
            images_list = sorted(os.listdir(path_Data + 'train/images/'))
            masks_list = sorted(os.listdir(path_Data + 'train/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'train/images/' + images_list[i]
                mask_path = path_Data + 'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer

            # 初始化增强器
            self.faint_augmentor = myFaintLesionAugmentor()
        else:
            images_list = sorted(os.listdir(path_Data + 'val/images/'))
            masks_list = sorted(os.listdir(path_Data + 'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data + 'val/images/' + images_list[i]
                mask_path = path_Data + 'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.test_transformer

    def set_phase(self, phase):
        """👑 动态切换训练阶段：调整 __len__ 的长度倍率"""
        self.phase = phase
        if self.train:
            if phase == 1:
                self.expand_ratio = 7
            elif phase == 2:
                self.expand_ratio = 1

    def __len__(self):
        return len(self.data) * self.expand_ratio

    def __getitem__(self, indx):
        real_idx = indx % len(self.data)
        aug_idx = indx // len(self.data)

        img_path, mask_path = self.data[real_idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        label = np.array(Image.open(mask_path).convert('L'))

        label = np.expand_dims(label, axis=-1)
        label = (label > 127).astype(np.float32)

        if self.train:
            if self.phase == 1 and aug_idx > 0:
                # Phase 1：高强度探索，必定生成特定的 6 种变体
                mode_map = {1: 'diffuse', 2: 'donut', 3: 'nested', 4: 'multi', 5: 'shattered', 6: 'ghost'}
                current_mode = mode_map.get(aug_idx, 'diffuse')
                image, label = self.faint_augmentor.apply_specific_mode((image, label), mode=current_mode)

            elif self.phase == 2:
                # 👑 Phase 2 修复：在 1N 数据集中，依然保留 25% 的概率随机生成一种极其微弱的变体
                # 这样 Decoder 就绝对不敢屏蔽 Encoder 传来的浅色特征了
                if random.random() < 0.50:
                    random_mode = random.choice(['diffuse', 'donut', 'shattered', 'ghost'])
                    image, label = self.faint_augmentor.apply_specific_mode((image, label), mode=random_mode)

            image, label = self.transformer((image, label))
        else:
            image, label = self.transformer((image, label))

        return image, label.long()


class Synapse_dataset(Dataset):
    """(保留原有 Synapse 代码，不做改动)"""

    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name + '.npz')
            data = np.load(data_path)
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy".format(vol_name)
            data = np.load(filepath)
            image, label = data[:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample