"""
定制版：医学图像数据集安全合并工具（解决不同命名规则）
"""

import os
import shutil
from pathlib import Path

# ==========================================
# 你的确切路径（已为你填好）
# ==========================================
# 数据集 A (原 isic18 训练集，命名规则：同名 .png)
DIR_A_IMG = Path(r"/root/root/VM-UNet/data/isic18/train/images")
DIR_A_MASK = Path(r"/root/root/VM-UNet/data/isic18/train/masks")

# 数据集 B (你新筛选出的数据，命名规则：_segmentation.png)
DIR_B_IMG = Path(r"/root/root/VM-UNet/TestData/filtered_results/images")
DIR_B_MASK = Path(r"/root/root/VM-UNet/TestData/filtered_results/masks")

# 合并后的新大本营
MERGED_DIR = Path(r"D:\HUBU.zhan\GDM\VM-UNet-main\TrainData_Merged")
MERGED_IMG = MERGED_DIR / "images"
MERGED_MASK = MERGED_DIR / "masks"

MERGED_IMG.mkdir(parents=True, exist_ok=True)
MERGED_MASK.mkdir(parents=True, exist_ok=True)
# ==========================================

def merge_dataset_a(prefix="src1_"):
    """处理数据集 A：原图和 Mask 都是 .png 且名字完全一样"""
    if not DIR_A_IMG.exists() or not DIR_A_MASK.exists():
        print("[报错] 找不到数据集 A 的路径！")
        return 0

    count = 0
    for img_path in DIR_A_IMG.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in {'.png', '.jpg'}:
            # A 数据集的 Mask 寻找规则：直接用原图的名字换成 .png 后缀
            mask_name = img_path.stem + ".png"
            mask_path = DIR_A_MASK / mask_name

            if mask_path.exists():
                shutil.copy2(img_path, MERGED_IMG / f"{prefix}{img_path.name}")
                shutil.copy2(mask_path, MERGED_MASK / f"{prefix}{mask_name}")
                count += 1
            else:
                print(f"[警告] 数据集A 缺失 Mask: {mask_name}")
    return count

def merge_dataset_b(prefix="src2_"):
    """处理数据集 B：原图是 .jpg，Mask 是带 _segmentation.png 后缀"""
    if not DIR_B_IMG.exists() or not DIR_B_MASK.exists():
        print("[报错] 找不到数据集 B 的路径！")
        return 0

    count = 0
    for img_path in DIR_B_IMG.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in {'.png', '.jpg'}:
            # B 数据集的 Mask 寻找规则：加上 _segmentation 后缀
            mask_name = img_path.stem + "_segmentation.png"
            mask_path = DIR_B_MASK / mask_name

            if mask_path.exists():
                shutil.copy2(img_path, MERGED_IMG / f"{prefix}{img_path.name}")
                shutil.copy2(mask_path, MERGED_MASK / f"{prefix}{mask_name}")
                count += 1
            else:
                print(f"[警告] 数据集B 缺失 Mask: {mask_name}")
    return count

if __name__ == "__main__":
    print(f"开始安全合并数据集...\n")

    print("正在处理数据集 A (不带后缀规则) ...")
    count_a = merge_dataset_a(prefix="isic18_")

    print("正在处理数据集 B (带 _segmentation 规则) ...")
    count_b = merge_dataset_b(prefix="new_")

    print("\n" + "="*40)
    print(f"🎉 成功解决规则冲突，合并大功告成！")
    print(f"-> 数据集 A (isic18): {count_a} 对")
    print(f"-> 数据集 B (filtered):  {count_b} 对")
    print(f"-> 总计可用训练集:    {count_a + count_b} 对")
    print(f"已安全保存至: {MERGED_DIR}")
    print("="*40)