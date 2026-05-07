"""
Consistency Reinforcement Loss
================================
基于 ConDSeg (AAAI'25) 的一致性损失，但适配 VM-UNet 的输出（已 sigmoid）。

ConDSeg paper Eq.6:
    L_cons(M1, M2) = 0.5 * ( BCE(B(M2,t), M1) + BCE(B(M1,t), M2) )

我们的 M1, M2 都已经过 sigmoid（VMUNet.forward 末尾有 torch.sigmoid），
所以可以直接当概率使用。

Two key implementation details:
1. 二值化的那一边必须 detach，避免梯度通过二值化路径回传（二值化不可导，
   detach 后等价于 stop-gradient + 提供伪标签的训练范式）。
2. 数值稳定：BCE 对接近 0/1 的 prob 数值不稳定，clamp 一下。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """
    Consistency Reinforcement loss.

    Args:
        threshold: 二值化阈值 (default 0.5)
        eps: 用于 clamp 概率，避免 log(0). (default 1e-7)

    Forward:
        M1, M2: 都是 sigmoid 后的概率张量, shape (B, 1, H, W) 或 (B, H, W).
                取值范围 [0, 1].

    Returns:
        scalar loss, 范围 [0, +inf)
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, M1: torch.Tensor, M2: torch.Tensor) -> torch.Tensor:
        # 二值化伪标签（stop-gradient）
        M1_bin = (M1 >= self.threshold).float().detach()
        M2_bin = (M2 >= self.threshold).float().detach()

        # 数值稳定的 BCE
        M1_clamped = M1.clamp(self.eps, 1.0 - self.eps)
        M2_clamped = M2.clamp(self.eps, 1.0 - self.eps)

        # 用 M1 当 target 监督 M2，再用 M2 当 target 监督 M1，平均
        loss = 0.5 * (
            F.binary_cross_entropy(M2_clamped, M1_bin) +
            F.binary_cross_entropy(M1_clamped, M2_bin)
        )
        return loss