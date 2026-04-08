import os
import torch
import numpy as np
import cv2
from PIL import Image

# 导入模型
from models.vmunet.vmunetff import VMUNet

# ==========================================
# ⚙️ 配置参数
# ==========================================
# 👑 强烈建议：将这里的路径替换为你最新训练出来的带有 miou 后缀的最佳权重
# 例如: 'checkpoints/best-epoch76-miou0.8288.pth'
MODEL_PATH = '/root/root/VM-UNet/results/vmunet_isic18_Monday_06_April_2026_00h_30m_14s/checkpoints/best.pth'
IMG_SIZE = 256
INPUT_DIR = 'inputs'
OUTPUT_MASK_DIR = 'outputs_mask'       # 保存用于指标计算的纯净二值 Mask
OUTPUT_OVERLAY_DIR = 'outputs_overlay' # 保存用于论文展示的直观叠加图

# 👑 强烈建议：将这里的 0.5 换成你测试阶段网格搜索出的最佳阈值
THRESHOLD = 0.5

# ISIC18 测试集固定均值和方差
ISIC18_TEST_MEAN = 149.034
ISIC18_TEST_STD = 32.022

# ==========================================
# 🛠️ 预处理函数
# ==========================================
def preprocess_exact_author_logic(img_pil, target_size):
    """复刻原作者数据流的标准化策略"""
    img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
    img = np.array(img_pil, dtype=np.float32)

    # 针对 ISIC 的特殊标准化后重新拉伸策略
    img_normalized = (img - ISIC18_TEST_MEAN) / ISIC18_TEST_STD
    img_min = np.min(img_normalized)
    img_max = np.max(img_normalized)

    if img_max > img_min:
        img_final = ((img_normalized - img_min) / (img_max - img_min)) * 255.0
    else:
        img_final = img_normalized

    img_tensor = torch.from_numpy(img_final).permute(2, 0, 1).contiguous().float()
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# ==========================================
# 🚀 主推理逻辑
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 确保输出目录存在
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OVERLAY_DIR, exist_ok=True)

    # 1. 初始化模型 (参数需与带有 ResPath 的结构对齐)
    model = VMUNet(
        num_classes=1,
        input_channels=3,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 2, 2, 1],
        drop_path_rate=0.2,
        load_ckpt_path=None
    ).to(device)

    # 2. 严谨地加载权重
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到权重文件，请检查路径: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else (
        checkpoint['model'] if 'model' in checkpoint else checkpoint)

    # 剔除可能存在的统计参数
    new_state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ 权重加载成功！包含最新 Lightweight ResPath 特征。")

    # 3. 开启评估模式
    model.eval()

    image_list = os.listdir(INPUT_DIR)
    valid_images = [img for img in image_list if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🚀 开始进行正式推理，共 {len(valid_images)} 张图片...")

    # 4. 执行推理
    with torch.no_grad():
        for img_name in valid_images:
            img_path = os.path.join(INPUT_DIR, img_name)
            img_pil = Image.open(img_path).convert('RGB')
            original_size = img_pil.size  # (Width, Height)

            img_tensor = preprocess_exact_author_logic(img_pil, IMG_SIZE).to(device)
            output = model(img_tensor)

            if isinstance(output, tuple):
                output = output[0]

            # 剥离 batch 维度，获得 256x256 的【浮点数概率图】
            prob_map = output.squeeze().cpu().numpy()

            # ----------------------------------------
            # 🌟 核心优化：先放大平滑的概率图，再做切刀！
            # ----------------------------------------
            # 1. 用双线性插值把浮点概率图恢复到原图尺寸 (边界非常丝滑)
            prob_map_resized = cv2.resize(prob_map, original_size, interpolation=cv2.INTER_LINEAR)

            # 2. 在高分辨率下使用最佳阈值进行二值化
            prediction = (prob_map_resized > THRESHOLD).astype(np.uint8) * 255

            # 3. 保存纯净 Mask
            mask_save_path = os.path.join(OUTPUT_MASK_DIR, os.path.splitext(img_name)[0] + '_pred.png')
            cv2.imwrite(mask_save_path, prediction)

            # ----------------------------------------
            # 🎨 论文可视化优化：生成病灶区域叠加图 (Overlay)
            # ----------------------------------------
            # 将 PIL 原图转换为 BGR 格式的 OpenCV 图像
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 创建红色蒙版 (对预测出病灶的区域标红)
            color_mask = np.zeros_like(img_cv)
            color_mask[prediction == 255] = [0, 0, 255]  # BGR 格式

            # 将红色蒙版以半透明形式叠加到原图上
            overlay = cv2.addWeighted(img_cv, 0.7, color_mask, 0.5, 0)

            # 提取边界轮廓并绘制，让病灶边缘更加锐利可见
            contours, _ = cv2.findContours(prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)  # 画粗度为 2 的红色轮廓

            # 保存叠加图
            overlay_save_path = os.path.join(OUTPUT_OVERLAY_DIR, os.path.splitext(img_name)[0] + '_overlay.jpg')
            cv2.imwrite(overlay_save_path, overlay)

            print(f"✔ 成功处理: {img_name}")

    print(f"🎉 批量推理完毕！\n纯净 Mask 已存入: {OUTPUT_MASK_DIR}\n可视化叠加图已存入: {OUTPUT_OVERLAY_DIR}")

if __name__ == '__main__':
    main()