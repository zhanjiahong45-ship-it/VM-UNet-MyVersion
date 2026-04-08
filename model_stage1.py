import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from models.vmunet.vmunetff import VMUNet


class MinimalistHead(nn.Module):
    """
    极简预测头：剥离高容量解码器，倒逼 Encoder 学习抗干扰能力
    """

    def __init__(self, in_channels_list=[96, 192, 384, 768]):
        super().__init__()
        # 将 Encoder 四层输出统一压缩到 64 通道 (掐住信息瓶颈)
        self.compress_convs = nn.ModuleList([
            nn.Conv2d(c, 64, kernel_size=1) for c in in_channels_list
        ])
        # 拼接后的 4*64=256 通道，用 1x1 卷积直接吐出预测概率
        self.final_conv = nn.Conv2d(64 * 4, 1, kernel_size=1)

    def forward(self, skip_list, target_size):
        upsampled_features = []
        for i, feat in enumerate(skip_list):
            # VM-UNet 特征是 [B, H, W, C]，需转成 [B, C, H, W]
            feat = feat.permute(0, 3, 1, 2).contiguous()
            feat = self.compress_convs[i](feat)
            # 暴力上采样回原图尺寸
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)

        # 拼接并输出
        out = torch.cat(upsampled_features, dim=1)
        out = self.final_conv(out)
        return torch.sigmoid(out)


class VMUNet_Stage1(nn.Module):
    """
    Stage 1 训练核心：FCD -> 纯净特征 -> (分支A/分支B增强) -> Encoder -> Minimalist Head
    """

    def __init__(self, base_model: VMUNet):
        super().__init__()
        self.vmunet = base_model.vmunet  # 取出底层的 VSSM 主干
        self.minimalist_head = MinimalistHead(in_channels_list=self.vmunet.dims)

        # 定义 Kornia 在线强增强 (在 GPU 上进行，不破坏梯度)
        self.strong_aug = K.AugmentationSequential(
            K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
            K.RandomGrayscale(p=0.5),
            K.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 3.0), p=0.5),
            data_keys=["input"]
        )

    def _forward_encoder(self, x):
        """仅执行 Encoder 部分提取多尺度特征"""
        skip_list = []
        if self.vmunet.ape:
            x = x + self.vmunet.absolute_pos_embed
        x = self.vmunet.pos_drop(x)
        for layer in self.vmunet.layers:
            skip_list.append(x)
            x = layer(x)
        return skip_list

    def forward(self, x):
        B, C, H, W = x.shape
        fcd = self.vmunet.patch_embed

        # ================== 第一道防线：物理清理 ==================
        # 调用你在 FCD 中写好的魔法函数，得到无毛发、色彩解耦的安全底图
        x_cleaned = fcd.clean_phase(x)
        x_enhanced = fcd.enhance_phase(x_cleaned)

        # ================== 核心分流点 ==================
        # 分支 A：干净流 (Teacher)
        x_patched_clean = fcd.proj(x_enhanced).permute(0, 2, 3, 1).contiguous()
        if fcd.norm is not None: x_patched_clean = fcd.norm(x_patched_clean)

        skip_list_clean = self._forward_encoder(x_patched_clean)
        mask_pred_clean = self.minimalist_head(skip_list_clean, (H, W))

        # 分支 B：强干扰流 (Student) - 第二道防线启动
        # 在 FCD 增强后的图上，故意施加破坏性软噪声
        x_enhanced_aug = self.strong_aug(x_enhanced)

        x_patched_aug = fcd.proj(x_enhanced_aug).permute(0, 2, 3, 1).contiguous()
        if fcd.norm is not None: x_patched_aug = fcd.norm(x_patched_aug)

        skip_list_aug = self._forward_encoder(x_patched_aug)
        mask_pred_aug = self.minimalist_head(skip_list_aug, (H, W))

        # Stage 1 返回两个预测结果，用于计算一致性
        return mask_pred_clean, mask_pred_aug