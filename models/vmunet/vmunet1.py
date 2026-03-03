import torch
from torch import nn
import torch.nn.functional as F
from .vmamba1 import VSSM, filter_pretrained_weights  # 导入过滤函数


class FocalLoss(nn.Module):
    """新增：Focal Loss解决难样本挖掘"""

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        BCE_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss


class DiceLoss(nn.Module):
    """Dice Loss基础实现"""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() + self.smooth
        return 1 - (2. * intersection + self.smooth) / union


class FocalDiceLoss(nn.Module):
    """组合损失：Focal + Dice（适配ISIC数据集）"""

    def __init__(self, wf=0.3, wd=0.7):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.wf = wf
        self.wd = wd

    def forward(self, pred, target):
        return self.wf * self.focal(pred, target) + self.wd * self.dice(pred, target)


class VMUNet1(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 2, 2],
                 depths_decoder=[2, 2, 2, 1],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,  # 必须保留，用于接收预训练权重路径
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits, fcd_out = self.vmunet(x)  # 获取分割结果+FCD输出
        if self.num_classes == 1:
            return torch.sigmoid(logits), fcd_out
        else:
            return logits, fcd_out

    def load_from(self):
        """
        终极修复：强制过滤不匹配参数，直接构建新的state_dict
        """
        if self.load_ckpt_path is not None:
            # 1. 加载当前模型的state_dict
            model_dict = self.vmunet.state_dict()
            # 2. 加载预训练权重
            modelCheckpoint = torch.load(self.load_ckpt_path,
                                         map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            pretrained_dict = modelCheckpoint['model'] if 'model' in modelCheckpoint else modelCheckpoint

            # 3. 强制过滤：只保留形状匹配的参数，彻底跳过不匹配的
            new_state_dict = {}
            skip_count = 0
            load_count = 0
            for k in model_dict.keys():
                # 跳过PatchEmbed和SS2D的不匹配参数
                if 'patch_embed.proj' in k or 'A_logs' in k or 'Ds' in k:
                    new_state_dict[k] = model_dict[k]  # 保留当前模型的参数
                    skip_count += 1
                    continue
                # 只加载存在且形状匹配的参数
                if k in pretrained_dict and model_dict[k].shape == pretrained_dict[k].shape:
                    new_state_dict[k] = pretrained_dict[k]
                    load_count += 1
                else:
                    # 保留当前模型的参数（随机初始化）
                    new_state_dict[k] = model_dict[k]
                    skip_count += 1

            # 4. 加载过滤后的state_dict（strict=True也不会报错）
            self.vmunet.load_state_dict(new_state_dict, strict=True)

            print(f"权重加载完成：成功加载 {load_count} 个参数，跳过 {skip_count} 个不匹配参数")
            print("V-Mamba weights loading finished (compatible mode)!")