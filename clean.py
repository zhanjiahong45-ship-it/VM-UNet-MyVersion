import os
import torch
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import save_image


class HairCleaner(torch.nn.Module):
    """
    独立提取的物理清洗层 (Physical Cleaning Phase)
    用于提取毛发 mask 并通过闭运算进行底色回填
    """
    def __init__(self, morph_kernel=5):
        super().__init__()
        self.morph_kernel = morph_kernel
        self.padding = self.morph_kernel // 2

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 【全局图像 Min-Max 归一化】保护色调一致性
        # 将 view 替换为 reshape 解决内存不连续报错
        x_min = x.reshape(B, -1).min(dim=-1)[0].reshape(B, 1, 1, 1)
        x_max = x.reshape(B, -1).max(dim=-1)[0].reshape(B, 1, 1, 1)
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)

        # 2. 找毛发位置 (在单通道灰度图上进行)
        x_gray = x_norm.mean(dim=1, keepdim=True)
        dilation_gray = F.max_pool2d(x_gray, kernel_size=self.morph_kernel, stride=1, padding=self.padding)
        erosion_gray = -F.max_pool2d(-dilation_gray, kernel_size=self.morph_kernel, stride=1, padding=self.padding)
        hair_mask = torch.clamp(torch.abs(x_gray - erosion_gray) * 10.0, 0.0, 1.0)

        # 3. 准备回填皮肤 (在三通道RGB图上进行闭运算)
        dilation_rgb = F.max_pool2d(x_norm, kernel_size=self.morph_kernel, stride=1, padding=self.padding)
        erosion_rgb = -F.max_pool2d(-dilation_rgb, kernel_size=self.morph_kernel, stride=1, padding=self.padding)

        # 4. 物理彩色回填
        x_cleaned = x_norm * (1 - hair_mask) + erosion_rgb * hair_mask

        return x_cleaned.clamp(1e-5, 1.0), hair_mask


if __name__ == "__main__":
    # ================= 你的路径配置 =================
    INPUT_PATH = r"D:\HUBU.zhan\GDM\VM-UNet-main\inputs"
    OUTPUT_PATH = r"D:\HUBU.zhan\GDM\VM-UNet-main\outputEE"

    # 支持的图片格式
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 1. 自动创建输出文件夹（如果不存在）
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 2. 自动检测 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 3. 初始化清洗器并放到对应设备上
    cleaner = HairCleaner(morph_kernel=5).to(device)

    # 4. 获取所有图片文件
    if not os.path.exists(INPUT_PATH):
        print(f"错误：找不到输入文件夹 {INPUT_PATH}，请检查路径。")
    else:
        image_files = [f for f in os.listdir(INPUT_PATH) if f.lower().endswith(VALID_EXTENSIONS)]

        if not image_files:
            print(f"在 {INPUT_PATH} 中没有找到支持的图片文件。")
        else:
            print(f"找到 {len(image_files)} 张图片，开始处理...")

            # 5. 遍历处理每张图片
            for filename in image_files:
                img_path = os.path.join(INPUT_PATH, filename)
                save_path = os.path.join(OUTPUT_PATH, filename)

                try:
                    # 读取图片 (格式为 [C, H, W], 数值 0-255) -> 转为浮点数 -> 归一化到 0-1 -> 增加 Batch 维度
                    img_tensor = read_image(img_path).float().unsqueeze(0) / 255.0
                    img_tensor = img_tensor.to(device)

                    # 运行清洗
                    with torch.no_grad():
                        cleaned_img, _ = cleaner(img_tensor)

                    # 保存结果 (save_image 会自动处理 batch 维度和范围)
                    save_image(cleaned_img, save_path)
                    print(f"成功处理并保存: {filename}")

                except Exception as e:
                    print(f"处理 {filename} 时发生错误: {str(e)}")

            print(f"全部处理完成！文件已保存至: {OUTPUT_PATH}")