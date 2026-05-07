import torch
import torch.nn as nn
import math
from typing import List
SPIRAL_KEYS = (
    'x_proj_spiral', 'dt_projs_weight_spiral', 'dt_projs_bias_spiral',
    'A_logs_spiral', 'Ds_spiral', 'spiral_alpha',
)

class LoRALinear(nn.Module):
    # （保留你原来的 LoRALinear 代码即可）
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters(): p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        device, dtype = base_linear.weight.device, base_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(r, base_linear.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r, device=device, dtype=dtype))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base(x) + self.scaling * (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T)

class LoRAConv2d1x1(nn.Module):
    """专门为 final_conv (1x1卷积) 准备的 LoRA 包装器"""
    def __init__(self, base_conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(base_conv, nn.Conv2d) and base_conv.kernel_size == (1, 1)
        self.base = base_conv
        for p in self.base.parameters(): p.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        device, dtype = base_conv.weight.device, base_conv.weight.dtype

        self.lora_A = nn.Conv2d(base_conv.in_channels, r, kernel_size=1, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Conv2d(r, base_conv.out_channels, kernel_size=1, bias=False, device=device, dtype=dtype)
        self.lora_dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))

def _get_parent_module(model: nn.Module, full_name: str):
    parts = full_name.split('.')
    parent = model
    for p in parts[:-1]: parent = getattr(parent, p)
    return parent, parts[-1]

def inject_lora_decoder(model: nn.Module, r: int = 8, alpha: int = 16, dropout: float = 0.0):
    """只将 LoRA 注入到 Decoder 部分的 out_proj 和 final_conv"""
    replaced = []

    to_replace_linear = []
    to_replace_conv = []

    for full_name, module in model.named_modules():
        # 目标 1: Decoder (layers_up) 中的 out_proj
        if 'layers_up' in full_name and full_name.endswith('out_proj') and isinstance(module, nn.Linear):
            to_replace_linear.append(full_name)
        # 目标 2: 整个网络的 final_conv
        elif full_name.endswith('final_conv') and isinstance(module, nn.Conv2d):
            to_replace_conv.append(full_name)

    # 注入 Linear
    for full_name in to_replace_linear:
        parent, attr = _get_parent_module(model, full_name)
        setattr(parent, attr, LoRALinear(getattr(parent, attr), r=r, alpha=alpha, dropout=dropout))
        replaced.append(full_name)

    # 注入 Conv2d
    for full_name in to_replace_conv:
        parent, attr = _get_parent_module(model, full_name)
        setattr(parent, attr, LoRAConv2d1x1(getattr(parent, attr), r=r, alpha=alpha, dropout=dropout))
        replaced.append(full_name)

    return replaced

def freeze_for_lora_finetune(model: nn.Module):
    # 1. 彻底冻结所有参数 (包含 Encoder、螺旋等)
    for p in model.parameters(): p.requires_grad = False

    # 2. 仅解冻 LoRA 参数
    trainable_count = 0
    for name, p in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
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
        f"Trainable layers: {len(trainable_names)}",
    ]
    # 只展示前 10 个名字，太多刷屏
    msg_lines.append("Sample trainable params:")
    for n, c in trainable_names[:10]:
        msg_lines.append(f"  - {n}  ({c:,})")
    if len(trainable_names) > 10:
        msg_lines.append(f"  ... and {len(trainable_names) - 10} more")

    msg = "\n".join(msg_lines)
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)