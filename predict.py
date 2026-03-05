import os
import torch
import numpy as np
import cv2
from PIL import Image

# 导入模型
from models.vmunet.vmunetff import VMUNet

# --- 配置参数 ---
MODEL_PATH = '/root/root/VM-UNet/results/vmunet_isic18_Wednesday_04_March_2026_21h_14m_52s/checkpoints/best-epoch95-loss0.3762.pth'
IMG_SIZE = 256
INPUT_DIR = 'inputs'
OUTPUT_DIR = 'outputO'

# 作者 utils.py 里写的 ISIC18 测试集固定均值和方差
ISIC18_TEST_MEAN = 149.034
ISIC18_TEST_STD = 32.022


def preprocess_exact_author_logic(img_pil, target_size):
    """
    100% 复刻作者 utils.py 里的 myNormalize 和 myToTensor
    """
    # 1. Resize
    img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)

    # 2. 转 Numpy (0-255)
    img = np.array(img_pil, dtype=np.float32)

    # 3. 复刻 myNormalize (先标准化，再拉伸回 0-255)
    img_normalized = (img - ISIC18_TEST_MEAN) / ISIC18_TEST_STD

    img_min = np.min(img_normalized)
    img_max = np.max(img_normalized)

    if img_max > img_min:
        # 作者的核心逻辑：标准化后，又强行拉伸到 0-255
        img_final = ((img_normalized - img_min) / (img_max - img_min)) * 255.0
    else:
        img_final = img_normalized

    # 4. 复刻 myToTensor (不除以255！)
    # HWC -> CHW
    # 【最小修复点】：在这里加一个 .contiguous() 强制内存连续，模型里就不会报 view 的错了！
    img_tensor = torch.from_numpy(img_final).permute(2, 0, 1).contiguous().float()
    img_tensor = img_tensor.unsqueeze(0)  # 加 batch 维度

    return img_tensor


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 初始化模型
    model = VMUNet(
        num_classes=1,
        input_channels=3,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 2, 2, 1],
        drop_path_rate=0.2,
        load_ckpt_path=None,
    ).to(device)

    # 加载权重
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        # 过滤掉统计参数
        new_state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
        model.load_state_dict(new_state_dict, strict=False)
        print("✅ 权重加载成功")
    else:
        print("❌ 找不到权重文件")
        return

    model.eval()

    image_list = os.listdir(INPUT_DIR)
    print(f"开始推理 {len(image_list)} 张图片...")

    with torch.no_grad():
        for img_name in image_list:
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(INPUT_DIR, img_name)
            img_pil = Image.open(img_path).convert('RGB')
            original_size = img_pil.size

            # --- 关键：使用复刻版预处理 ---
            img_tensor = preprocess_exact_author_logic(img_pil, IMG_SIZE).to(device)

            # --- 预测 ---
            # 因为模型里自带 sigmoid，所以输出直接就是 0-1 的概率
            output = model(img_tensor)

            # 这里的 output 已经是概率了，不需要再 sigmoid
            output = output.squeeze().cpu().numpy()

            # --- 保存结果 ---
            prediction = (output > 0.5).astype(np.uint8) * 255
            prediction = cv2.resize(prediction, original_size, interpolation=cv2.INTER_NEAREST)

            save_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + '_pred.png')
            cv2.imwrite(save_path, prediction)
            print(f"Saved: {save_path}")


if __name__ == '__main__':
    main()