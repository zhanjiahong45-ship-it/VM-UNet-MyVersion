import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryConsistencyLoss(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, mask1, mask2):
        # 核心细节：必须加上 .detach()！
        # 互为伪标签时，不能让梯度从 target 侧流走，否则模型会把预测推向全0或全1的坍塌解
        mask1_binary = (mask1 > self.threshold).float().detach()
        mask2_binary = (mask2 > self.threshold).float().detach()

        # 互相把对方的二值化结果当作 Ground Truth 来算交叉熵
        loss_1_to_2 = F.binary_cross_entropy(mask1, mask2_binary, reduction='mean')
        loss_2_to_1 = F.binary_cross_entropy(mask2, mask1_binary, reduction='mean')

        return loss_1_to_2 + loss_2_to_1