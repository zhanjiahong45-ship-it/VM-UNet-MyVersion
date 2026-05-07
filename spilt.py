import os
import random
import shutil
from pathlib import Path
from PIL import Image  # 引入 PIL 用于安全转换格式


def split_and_convert_dataset(img_dir, mask_dir, output_dir, split_ratio=(25, 7)):
    img_path = Path(img_dir)
    mask_path = Path(mask_dir)
    out_path = Path(output_dir)

    # 1. 获取所有图像文件
    images = sorted([f for f in os.listdir(img_path) if f.lower().endswith(('.png', '.jpg', '.bmp'))])
    masks = os.listdir(mask_path)

    random.seed(42)
    random.shuffle(images)

    total = len(images)
    split_idx = int(total * split_ratio[0] / sum(split_ratio))

    train_files = images[:split_idx]
    val_files = images[split_idx:]

    for split in ['train', 'val']:
        (out_path / split / 'image').mkdir(parents=True, exist_ok=True)
        (out_path / split / 'mask').mkdir(parents=True, exist_ok=True)

    def process_and_save(files, split_name):
        count = 0
        for f in files:
            name_stem = Path(f).stem
            # 统一输出文件名
            new_name = f"{name_stem}.png"

            # 查找对应的 mask
            target_mask_old = next((m for m in masks if m.startswith(name_stem)), None)

            if target_mask_old:
                # 处理并保存 Image
                with Image.open(img_path / f) as img:
                    img.save(out_path / split_name / 'image' / new_name, "PNG")

                # 处理并保存 Mask
                with Image.open(mask_path / target_mask_old) as msk:
                    msk.save(out_path / split_name / 'mask' / new_name, "PNG")

                count += 1
            else:
                print(f"⚠️ 跳过: 找不到 {name_stem} 的匹配 Mask")
        return count

    print("🚀 正在转换并划分数据...")
    t_count = process_and_save(train_files, 'train')
    v_count = process_and_save(val_files, 'val')

    print(f"\n✅ 处理完成！")
    print(f"训练集: {t_count} 对 | 验证集: {v_count} 对 (全部已统一为 .png)")


# --- 路径配置 ---
IMG_DIR = r'D:\HUBU.zhan\GDM\VM-UNet-main\ph2_dataset\image'
MASK_DIR = r'D:\HUBU.zhan\GDM\VM-UNet-main\ph2_dataset\mask'
SAVE_DIR = r'D:\HUBU.zhan\GDM\VM-UNet-main\ph2_dataset\split_data'

split_and_convert_dataset(IMG_DIR, MASK_DIR, SAVE_DIR)