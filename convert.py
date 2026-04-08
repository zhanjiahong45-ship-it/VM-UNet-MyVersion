import cv2
import numpy as np
import os


def single_image_erasure():
    # ==========================================
    # 1. 严格按照您提供的路径配置
    # ==========================================
    img_path = r"D:\HUBU.zhan\GDM\VM-UNet-main\inputs\2.png"
    mask_path = r"D:\HUBU.zhan\GDM\VM-UNet-main\inputsm\2.png"
    output_dir = r"D:\HUBU.zhan\GDM\VM-UNet-main\outputEE"

    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "12.png")

    # ==========================================
    # 2. 加载图像与掩码 (加入防爆错机制)
    # ==========================================
    if not os.path.exists(img_path):
        print(f"找不到原图，请检查路径: {img_path}")
        return
    if not os.path.exists(mask_path):
        print(f"找不到Mask，请检查路径: {mask_path}")
        return

    img = cv2.imread(img_path)
    # 掩码必须以单通道灰度图模式读取
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    print("图像和Mask加载成功，开始执行挖掘与修复...")

    # ==========================================
    # 3. 核心处理逻辑
    # ==========================================
    # 步骤A：确保Mask是纯粹的二值图 (0 和 255)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 步骤B：适度膨胀Mask (极其关键！)
    # iteration=3 意味着向外扩充大约3个像素，确保把黑斑周围的高梯度过渡带彻底覆盖
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=3)

    # 步骤C：Navier-Stokes 流体力学图像修复
    # inpaintRadius=5 决定了算法从空洞边缘向外参考多少像素的颜色来填补
    inpainted_img = cv2.inpaint(img, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)

    # ==========================================
    # 4. 输出结果
    # ==========================================
    cv2.imwrite(output_path, inpainted_img)
    print(f"处理完成！成功挖掉中心病灶并修复。")
    print(f"请前往查看结果: {output_path}")


if __name__ == "__main__":
    single_image_erasure()