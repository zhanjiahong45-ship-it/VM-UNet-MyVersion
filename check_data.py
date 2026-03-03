import os
from PIL import Image
import warnings

# 忽略一些安全的警告
warnings.filterwarnings("ignore")

# 你的数据集路径
DATA_DIR = '/root/root/VM-UNet/data/isic17/'
SPLITS = ['train', 'val']  # 如果有 test 也可以加上 'test'


def check_and_clean_dataset():
    corrupted_count = 0
    for split in SPLITS:
        img_dir = os.path.join(DATA_DIR, split, 'images')
        mask_dir = os.path.join(DATA_DIR, split, 'masks')  # 注意：请确认你的掩膜文件夹是不是叫 masks 或者 labels

        if not os.path.exists(img_dir):
            continue

        print(f"正在扫描 {split} 集...")
        img_names = os.listdir(img_dir)

        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)

            try:
                # verify() 比 open() 更快，专门用来检测图片文件是否完整
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                print(f"⚠️ 发现损坏的图片: {img_path} | 错误: {e}")

                # 1. 删除损坏的图片
                os.remove(img_path)
                print(f"✅ 已删除原图: {img_name}")

                # 2. 找到并删除对应的 Mask (处理后缀可能不一样的情况)
                # ISIC 数据集的 mask 通常叫 ISIC_XXXXXXX_segmentation.png 或者和原图同名
                base_name = os.path.splitext(img_name)[0]
                possible_mask_names = [
                    base_name + '.png',
                    base_name + '.jpg',
                    base_name + '_segmentation.png'
                ]

                for m_name in possible_mask_names:
                    mask_path = os.path.join(mask_dir, m_name)
                    if os.path.exists(mask_path):
                        os.remove(mask_path)
                        print(f"✅ 已同步删除 Mask: {m_name}")
                        break

                corrupted_count += 1

    print(f"\n🎉 扫描完成！共清理了 {corrupted_count} 张损坏的图片及其标签。")


if __name__ == '__main__':
    check_and_clean_dataset()