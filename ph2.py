import cv2
import numpy as np
import glob

# 将这里的路径替换为你本地 PH2 数据集图片的实际路径
# 例如: 'PH2Dataset/PH2_Images/*/*_Dermoscopic_Image/*.bmp'
image_paths = glob.glob(r'D:\HUBU.zhan\GDM\VM-UNet-main\ph2_dataset\image\*.bmp')

pixels = []
for path in image_paths:
    # 假设你的 utils.py 原逻辑是单通道(灰度)图
    #img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 如果原网络是三通道(RGB)输入，请使用：
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    if img is not None:
        pixels.append(img.flatten())

pixels = np.concatenate(pixels)
print(f"PH2_MEAN = {np.mean(pixels):.3f}")
print(f"PH2_STD = {np.std(pixels):.3f}")