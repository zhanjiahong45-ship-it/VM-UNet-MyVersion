import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F

# ================= 1. 设置路径 =================
# 将输入路径改为你提供的图片
input_img_path = "/root/root/VM-UNet/inputs/22.png"
output_dir = "/root/root/VM-UNet/inputst"

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# ================= 2. 读取图像并转换为 Tensor =================
transform = transforms.ToTensor()

# 读取原图 x (保留 RGB 细节)
img_pil = Image.open(input_img_path).convert("RGB")
x_rgb = transform(img_pil).unsqueeze(0) # 形状: [1, 3, H, W]

# 转换为灰度图用于阈值计算
img_gray_pil = img_pil.convert("L")
x_gray = transform(img_gray_pil).unsqueeze(0) # 形状: [1, 1, H, W]

print(f"成功加载图像，Tensor 形状 (RGB): {x_rgb.shape}, (灰度): {x_gray.shape}")

# ================= 3. 执行修正后的核心逻辑 =================
# 基于图像特征微调阈值（范围 [0.0, 1.0]）
core_thresh = 0.45  # 找到中心核心（最暗区域）
halo_thresh = 0.65  # 找到核心和晕环的外部边界

# A. 提取区域掩码 (Spatial Masks)
# 核心区域 (Core): 像素值比 core_thresh 还要暗
mask_core = (x_gray < core_thresh)
# 目标区域 (Core + Halo): 像素值比 halo_thresh 暗（包括核心和晕环）
mask_total_target = (x_gray < halo_thresh)

# B. 几何方法：提取“甜甜圈 (Donut)”区域
# 纯晕环区域 = 目标区域 - 核心区域
mask_donut = mask_total_target & (~mask_core)

# C. 模拟 Spatial Router 的精准控制
# 我们创建一个热力图样式的权重图，其中只有提取出的环形区域权重最高
spatial_weight_halo = mask_donut.float() # [B, 1, H, W]

# ================= 4. 高对比度彩色融合 =================
# 为了“突出整个甜甜圈”，我们将 halo 区域融合为一种鲜艳的颜色。
# 核心和背景保持原图灰度细节，只把提取出的甜甜圈区域涂上橙色。

# 创建一个纯色（例如鲜橙色）的 Enhance 分支
# RGB (1.0, 0.5, 0.0)
enhanced_halo_rgb = torch.tensor([1.0, 0.5, 0.0]).view(1, 3, 1, 1).expand_as(x_rgb)

# 最终融合图
# 1. 保留原图：在非晕环区域（1.0 - weight）使用原图
fused_output_rgb = (1.0 - spatial_weight_halo) * x_rgb

# 2. 突出晕环：在晕环区域使用橙色
fused_output_rgb = fused_output_rgb + (spatial_weight_halo * enhanced_halo_rgb)

# ================= 5. 保存结果 =================
fused_save_path = os.path.join(output_dir, "3_fused_corrected.png")

# 保存图像 (去掉 Batch 维度)
save_image(fused_output_rgb.squeeze(0), fused_save_path)

print(f"✅ 测试完成！")
print(f" - 修正后的高对比度融合图（突出整个甜甜圈）已保存至: {fused_save_path}")
print(f" - 脚本使用了几何方法（目标-核心）来提取整个环形区域。")