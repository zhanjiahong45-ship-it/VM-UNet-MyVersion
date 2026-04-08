import os
import torch
import numpy as np
import cv2
from PIL import Image

# 导入模型
from models.vmunet.vmunetff import VMUNet

# ==========================================
# ⚙️ 配置参数 (记得确认权重路径)
# ==========================================
MODEL_PATH = '/root/root/VM-UNet/results/vmunet_isic18_Thursday_26_March_2026_21h_06m_13s/checkpoints/best.pth'
IMG_SIZE = 256
INPUT_DIR = 'inputs'
OUTPUT_DIR = 'output_debug'
THRESHOLD = 0.5

ISIC18_TEST_MEAN = 149.034
ISIC18_TEST_STD = 32.022


def preprocess_exact_author_logic(img_pil, target_size):
    img_pil = img_pil.resize((target_size, target_size), Image.BILINEAR)
    img = np.array(img_pil, dtype=np.float32)
    img_normalized = (img - ISIC18_TEST_MEAN) / ISIC18_TEST_STD
    img_min = np.min(img_normalized)
    img_max = np.max(img_normalized)
    if img_max > img_min:
        img_final = ((img_normalized - img_min) / (img_max - img_min)) * 255.0
    else:
        img_final = img_normalized
    img_tensor = torch.from_numpy(img_final).permute(2, 0, 1).contiguous().float()
    return img_tensor.unsqueeze(0)


def convert_to_binary_visual(tensor_prob, original_size):
    """将概率张量转为 0或255 的黑白图片，并转为BGR格式以便拼接"""
    prob_np = tensor_prob.squeeze().cpu().numpy()
    binary = (prob_np > THRESHOLD).astype(np.uint8) * 255
    binary = cv2.resize(binary, original_size, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    model = VMUNet(
        num_classes=1, input_channels=3, depths=[2, 2, 9, 2],
        depths_decoder=[2, 2, 2, 1], drop_path_rate=0.2, load_ckpt_path=None
    ).to(device)

    # 加载权重
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else (
        checkpoint['model'] if 'model' in checkpoint else checkpoint)
    new_state_dict = {k: v for k, v in state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ 权重加载成功！")

    model.eval()
    image_list = [img for img in os.listdir(INPUT_DIR) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🚀 开始极客版全景诊断，共 {len(image_list)} 张...")

    with torch.no_grad():
        for img_name in image_list:
            img_path = os.path.join(INPUT_DIR, img_name)
            img_pil = Image.open(img_path).convert('RGB')
            original_size = img_pil.size

            img_tensor = preprocess_exact_author_logic(img_pil, IMG_SIZE).to(device)

            # ==========================================================
            # ⭐ 核心：传入 debug=True，完美接住这两个 Mask
            # ==========================================================
            final_prob, mask0_prob = model(img_tensor, debug=True)

            # 转换为可视化图像
            mask0_img = convert_to_binary_visual(mask0_prob, original_size)
            final_mask_img = convert_to_binary_visual(final_prob, original_size)

            orig_cv2 = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # 打上红字标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(orig_cv2, 'Original', (10, 30), font, 1, (0, 255, 0), 2)
            cv2.putText(mask0_img, 'Mask0 (Locator)', (10, 30), font, 1, (0, 0, 255), 2)
            cv2.putText(final_mask_img, 'Mask Final (U-Net)', (10, 30), font, 1, (255, 0, 0), 2)

            # 水平拼接 3 张图
            combined_image = np.hstack((orig_cv2, mask0_img, final_mask_img))

            save_path = os.path.join(OUTPUT_DIR, os.path.splitext(img_name)[0] + '_overview.png')
            cv2.imwrite(save_path, combined_image)
            print(f"✔ 诊断概览图已保存: {save_path}")


if __name__ == '__main__':
    main()