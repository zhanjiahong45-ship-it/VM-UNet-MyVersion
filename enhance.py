import cv2
import numpy as np


def nothing(x):
    pass


# 1. 读取一张你觉得最不明显、最需要测试的图片 (替换为你真实的单张图片路径)
img_path = r"D:\HUBU.zhan\GDM\VM-UNet-main\inputs\img_1.png"
img = cv2.imread(img_path)

if img is None:
    print(f"❌ 找不到图片，请检查路径: {img_path}")
    exit()

# 创建交互窗口
cv2.namedWindow('Fast Image Tuner', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fast Image Tuner', 800, 600)

# 2. 创建滑动条 (Trackbars)
# OpenCV 滑动条不支持小数和负数，所以通过除以 10 或减去偏移量来实现
# L通道亮度拉扯 (Gamma): 范围 1-50 -> 对应 0.1 到 5.0 (默认25 -> 2.5)
cv2.createTrackbar('L_Gamma', 'Fast Image Tuner', 25, 50, nothing)

# L通道对比度 (CLAHE): 范围 0-100 -> 对应 0 到 10.0 (默认30 -> 3.0)
cv2.createTrackbar('L_CLAHE', 'Fast Image Tuner', 30, 100, nothing)

# R, G, B 颜色通道微调: 范围 0-200 -> 对应 -100 到 +100 (默认100 -> 不增不减)
cv2.createTrackbar('R_Offset', 'Fast Image Tuner', 100, 200, nothing)
cv2.createTrackbar('G_Offset', 'Fast Image Tuner', 100, 200, nothing)
cv2.createTrackbar('B_Offset', 'Fast Image Tuner', 100, 200, nothing)

print("✅ 调参工具已启动！")
print("👉 拖动滑动条实时查看效果。")
print("👉 按 'ESC' 键退出程序。")

while True:
    # 获取当前滑动条的值
    gamma = cv2.getTrackbarPos('L_Gamma', 'Fast Image Tuner') / 10.0
    gamma = max(0.1, gamma)  # 防止 gamma 为 0 报错
    clahe_limit = cv2.getTrackbarPos('L_CLAHE', 'Fast Image Tuner') / 10.0

    r_off = cv2.getTrackbarPos('R_Offset', 'Fast Image Tuner') - 100
    g_off = cv2.getTrackbarPos('G_Offset', 'Fast Image Tuner') - 100
    b_off = cv2.getTrackbarPos('B_Offset', 'Fast Image Tuner') - 100

    # === 第一阶段：L通道 (亮度/对比度) 处理 ===
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if clahe_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)

    l = np.power(l / 255.0, gamma) * 255.0
    l = np.clip(l, 0, 255).astype(np.uint8)

    merged_lab = cv2.merge((l, a, b))
    processed = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # === 第二阶段：RGB 颜色微调 ===
    # 转为 int16 防止加减法时数据溢出 (例如 250+10=4 而不是 260)
    processed = processed.astype(np.int16)

    # 注意：OpenCV 读取的默认通道顺序是 B, G, R
    processed[:, :, 0] += b_off  # Blue
    processed[:, :, 1] += g_off  # Green
    processed[:, :, 2] += r_off  # Red

    # 截断回 0-255 并转回 uint8 显示
    processed = np.clip(processed, 0, 255).astype(np.uint8)

    # 左右拼接对比显示 (左边原图，右边调参后)
    combined_display = np.hstack((img, processed))
    cv2.imshow('Fast Image Tuner', combined_display)

    # 按 ESC 键退出
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("\n=== 最终选定的参数 ===")
        print(f"L_Gamma  : {gamma}")
        print(f"L_CLAHE  : {clahe_limit}")
        print(f"R_Offset : {r_off}")
        print(f"G_Offset : {g_off}")
        print(f"B_Offset : {b_off}")
        break

cv2.destroyAllWindows()