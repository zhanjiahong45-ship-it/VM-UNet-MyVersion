import cv2
import os
import glob

input_dir = r"D:\HUBU.zhan\GDM\VM-UNet-main\inputs"
output_dir = r"D:\HUBU.zhan\GDM\VM-UNet-main\outputEE"
os.makedirs(output_dir, exist_ok=True)

# 抓取常见图片格式
image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + \
              glob.glob(os.path.join(input_dir, "*.png")) + \
              glob.glob(os.path.join(input_dir, "*.jpeg"))

for img_path in image_paths:
    # 1. 直接以灰度模式读取图片
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # 2. 最暴力的 Otsu 自动阈值分割 (反转黑白让病灶区域变白)
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. 直接保存到输出目录
    cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), mask)

print("极简预分割批量生成完毕！")