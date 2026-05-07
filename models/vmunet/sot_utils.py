import torch
import torch.nn as nn


class SOTLayerNorm(nn.Module):
    """
    SOT 注入器：包装 SS2D 中的 out_norm，
    实现在进入层归一化前，动态注入 State-offset (y')。
    这在数学上完全等价于 SOT(y) 的操作，且无需修改 vmambaff.py 原文件。
    """

    def __init__(self, base_norm: nn.LayerNorm, d_inner: int):
        super().__init__()
        self.base_norm = base_norm

        # 冻结原始的 LayerNorm 参数 (weight 和 bias)
        for p in self.base_norm.parameters():
            p.requires_grad = False

        # 新增 SOT(y) 的可训练参数 y'
        # 形状设为 (d_inner,)，利用 PyTorch 的广播机制自动作用于 (B, H, W, d_inner)
        device = next(base_norm.parameters()).device
        dtype = next(base_norm.parameters()).dtype
        self.state_offset_y = nn.Parameter(torch.zeros(d_inner, device=device, dtype=dtype))

    def forward(self, x):
        # x 的形状是 (B, H, W, d_inner)，即 5 条 SSM 路径融合并转置后的输出
        # 在进入 norm 前，加上 SOT 偏移量 y'
        x_sot = x + self.state_offset_y
        return self.base_norm(x_sot)


def inject_sot(model: nn.Module):
    """
    遍历模型，找到所有的 SS2D 模块，用 SOTLayerNorm 动态替换其 out_norm。
    """
    replaced_count = 0
    for name, module in model.named_modules():
        # 通过类名匹配，避免引入外部依赖
        if module.__class__.__name__ == 'SS2D':
            if hasattr(module, 'out_norm') and hasattr(module, 'd_inner'):
                base_norm = module.out_norm
                # 包装并替换
                sot_norm = SOTLayerNorm(base_norm, module.d_inner)
                module.out_norm = sot_norm
                replaced_count += 1

    print(f"[SOT Injector] Successfully injected State-offset Tuning into {replaced_count} SS2D blocks.")
    return replaced_count


def freeze_for_sot_finetune(model: nn.Module):
    """
    SOT 冻结策略：
    1) 默认全局冻结所有参数 requires_grad = False
    2) 仅解冻 'state_offset_y' (新增的 SOT 核心参数)
    3) 附带解冻 'spiral_alpha' (原模型中第5条路径的权重门控)
    """
    # Step 1: 全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # Step 2: 解冻核心微调参数
    trainable_count = 0
    for name, p in model.named_parameters():
        if 'state_offset_y' in name or 'spiral_alpha' in name:
            p.requires_grad = True
            trainable_count += 1

    return trainable_count


def report_trainable(model: nn.Module, logger=None):
    """打印可训练参数明细，用于 sanity check。"""
    total = 0
    trainable = 0
    trainable_names = []
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            trainable_names.append((name, n))

    msg_lines = [
        f"Total params:     {total:,}",
        f"Trainable params: {trainable:,}  ({100.0 * trainable / total:.4f}%)",
        f"Trainable tensors:{len(trainable_names)}",
        "-----------------------------------------",
        "Sample trainable params:"
    ]
    # 展示前 10 个和后 2 个，避免刷屏
    for n, c in trainable_names[:10]:
        msg_lines.append(f"  - {n}  ({c:,})")
    if len(trainable_names) > 10:
        msg_lines.append(f"  ... and {len(trainable_names) - 10} more")

    msg = "\n".join(msg_lines)
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)