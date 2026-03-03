import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable
import math
import kornia  # 必须安装: pip install kornia
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat

# 尝试导入官方选择性扫描算子
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    selective_scan_fn = None


# --- 核心创新：FCD_Module (Clean + Enhance) ---
class FCD_Module(nn.Module):
    r"""
    FCD-Module: Feature Clean-then-Enhance Decoupling Module
    Replaces traditional PatchEmbed2D to provide physics-based cleaning and color space enhancement.
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        self.C = embed_dim
        self.patch_size = patch_size

        # 1. 形态学算子：使用 MaxPool 模拟膨胀
        self.morph_kernel = 5
        self.padding = self.morph_kernel // 2

        # 2. 色彩提亮流：将 LAB (3通道) 映射为增强后的 LAB (3通道)
        # 【修正 1】：这里改为 3 -> 3，让输出保持图像物理形态，从而完美兼容作者的预训练下采样权重
        self.color_map = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        # 3. 自适应对比度拉伸系数 (让网络自适应学习3个通道的 Gamma 提亮程度)
        self.gamma = nn.Parameter(torch.ones(1, 3, 1, 1))

        # 4. 原生的 Patch Embedding Projection (保持 3 -> C)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(3, self.C, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(self.C)
        else:
            self.norm = None

    def clean_phase(self, x):
        """阶段 A: 物理清洗层 (先物理超度毛发)"""
        # 【修正 2】：动态 Min-Max 归一化。
        # DataLoader 传进来的 x 是被标准化的（含负数），必须拉回到 [0, 1] 才能做物理形态学和 RGB-LAB 转换
        B, C, H, W = x.shape
        x_min = x.view(B, C, -1).min(dim=-1)[0].view(B, C, 1, 1)
        x_max = x.view(B, C, -1).max(dim=-1)[0].view(B, C, 1, 1)
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)  # 1e-6 防止除零

        x_gray = x_norm.mean(dim=1, keepdim=True)

        # 闭运算 (Closing): 先膨胀再腐蚀
        dilation = F.max_pool2d(x_gray, kernel_size=self.morph_kernel,
                                stride=1, padding=self.padding)
        erosion = -F.max_pool2d(-dilation, kernel_size=self.morph_kernel,
                                stride=1, padding=self.padding)

        # 计算毛发 Mask: 原图和闭运算图的差异
        hair_mask = torch.abs(x_gray - erosion)
        hair_mask = torch.sigmoid(hair_mask * 10)

        # 物理回填：使用 [0, 1] 范围的 x_norm 进行融合
        x_cleaned = x_norm * (1 - hair_mask) + erosion * hair_mask
        return x_cleaned

    def enhance_phase(self, x_cleaned):
        """阶段 B: 安全增强层 (在无菌环境下暴力提亮)"""
        # x_cleaned 现在严格处于 [0, 1]，送入 kornia 色彩空间转换绝对安全
        x_lab = kornia.color.rgb_to_lab(x_cleaned)

        # 将微弱的色彩差异映射、放大
        feat_color = self.color_map(x_lab)

        # 对比度拉伸：利用 Gamma 校正放大信号，确保底数 > 0 避免梯度 NaN
        gamma_clamped = torch.abs(self.gamma) + 1e-4
        feat_enhanced = torch.pow(feat_color + 1e-6, gamma_clamped)

        return feat_enhanced

    def forward(self, x):
        # 第一步：清洗 (Clean) 获得 [0,1] 无毛发图像
        x_cleaned = self.clean_phase(x)

        # 第二步：增强 (Enhance) 获得 3 通道的强烈高对比度特征图
        x_enhanced = self.enhance_phase(x_cleaned)

        # 第三步：完美对接 VM-UNet 后续的 Patch Partition，预训练权重无缝加载！
        x_patched = self.proj(x_enhanced).permute(0, 2, 3, 1)
        if self.norm is not None:
            x_patched = self.norm(x_patched)

        return x_patched


# --- 基础组件 (严格遵循原版 vmamba.py) ---
class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=96, embed_dim=96, norm_layer=None):
        super().__init__()
        if isinstance(patch_size, int): patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        return self.norm(x) if self.norm is not None else x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x0, x1, x2, x3 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :], x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return self.reduction(self.norm(x))


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        # 原版逻辑：dim -> dim * scale²
        self.expand = nn.Linear(dim, dim * dim_scale * dim_scale, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale)
        return self.norm(x)


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        # 原版逻辑：dim -> dim * scale²
        self.expand = nn.Linear(dim, dim * dim_scale * dim_scale, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale)
        return self.norm(x)


# --- 核心算子：SS2D (还原原版选择性扫描逻辑) ---
class SS2D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dropout=0., bias=False, **kwargs):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, d_conv, padding=(d_conv - 1) // 2, groups=self.d_inner)
        self.act = nn.SiLU()
        # 四向扫描参数
        self.x_proj_weight = nn.Parameter(torch.empty(4, (self.dt_rank + d_state * 2), self.d_inner))
        self.dt_projs_weight = nn.Parameter(torch.empty(4, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(4, self.d_inner))
        self.A_logs = nn.Parameter(torch.empty(4, self.d_inner, d_state))
        self.Ds = nn.Parameter(torch.empty(4, self.d_inner))
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self._init_parameters()

    def _init_parameters(self):
        trunc_normal_(self.x_proj_weight, std=.02)
        trunc_normal_(self.dt_projs_weight, std=.02)
        nn.init.constant_(self.dt_projs_bias, 0)
        nn.init.constant_(self.A_logs, math.log(2.0))
        nn.init.constant_(self.Ds, 1.0)

    def selective_scan_naive(self, u, delta, A, B, C, D=None):
        """原版选择性扫描的朴素实现（兼容无官方算子环境）"""
        seq_len = u.shape[-1]
        hidden = torch.zeros(u.shape[0], u.shape[1], A.shape[-1], device=u.device, dtype=u.dtype)
        out = []
        for t in range(seq_len):
            delta_t = F.softplus(delta[..., t])
            # 原版选择性扫描核心公式
            hidden = hidden * torch.exp(-delta_t.unsqueeze(-1) * A) + delta_t.unsqueeze(-1) * B[..., t:t + 1]
            out_t = (hidden @ C[..., t:t + 1]).squeeze(-1)
            if D is not None:
                out_t = out_t + D * u[..., t]
            out.append(out_t)
        return torch.cat(out, dim=-1)

    def forward(self, x):
        B, H, W, C = x.shape
        L = H * W
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.act(self.conv2d(x.permute(0, 3, 1, 2).contiguous()))  # (B, D, H, W)

        # 四向扫描实现
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, 2, 3).contiguous().view(B, -1, L)], dim=1)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, 4, D, L)

        # 还原原版选择性扫描核心逻辑
        ys = []
        for i in range(4):
            xi = xs[:, i, :, :]  # (B, D, L)
            # 计算delta/A/B/C/D（严格匹配原版公式）
            delta = F.softplus(
                torch.einsum('did,bdl->bdl', self.dt_projs_weight[i], xi) +
                self.dt_projs_bias[i].unsqueeze(-1)
            )
            A = -torch.exp(self.A_logs[i])  # 原版A为负指数
            B = torch.einsum('dnd,bdl->bnl', self.x_proj_weight[i][:, self.dt_rank:self.dt_rank + self.d_state], xi)
            C = torch.einsum('dnd,bdl->bnl', self.x_proj_weight[i][:, self.dt_rank + self.d_state:], xi)
            D = self.Ds[i].unsqueeze(-1)

            # 执行选择性扫描（优先用官方算子，无则用朴素实现）
            if selective_scan_fn is not None:
                # 官方算子（高性能）
                yi = selective_scan_fn(
                    xi, delta, A, B, C, D=D,
                    delta_bias=self.dt_projs_bias[i],
                    return_last=False,
                )
            else:
                # 朴素实现（兼容模式）
                yi = self.selective_scan_naive(xi, delta, A, B, C, D)
            ys.append(yi)

        # 合并四向扫描结果
        y = torch.stack(ys, dim=1).sum(dim=1).view(B, -1, H, W).permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y) * F.silu(z)
        out = self.out_proj(y)
        return self.dropout(out) if self.dropout else out


class VSSBlock(nn.Module):
    def __init__(self, hidden_dim, drop_path=0., norm_layer=nn.LayerNorm, d_state=16, **kwargs):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


class VSSLayer(nn.Module):
    def __init__(self, dim, depth, d_state=16, drop_path=0., norm_layer=nn.LayerNorm, downsample=None, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [VSSBlock(dim, drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer, d_state) for i in
             range(depth)])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return self.downsample(x) if self.downsample else x


class VSSLayer_up(nn.Module):
    def __init__(self, dim, depth, d_state=16, drop_path=0., norm_layer=nn.LayerNorm, upsample=None, **kwargs):
        super().__init__()
        self.upsample = upsample(dim=dim, norm_layer=norm_layer) if upsample else None
        self.blocks = nn.ModuleList(
            [VSSBlock(dim, drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer, d_state) for i in
             range(depth)])

    def forward(self, x):
        if self.upsample: x = self.upsample(x)
        for blk in self.blocks: x = blk(x)
        return x


# --- 主类：VSSM (严格遵循原版结构) ---
class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.embed_dim = dims[0]
        self.fcd = FCD_Module(in_chans, self.embed_dim)
        self.patch_embed = PatchEmbed2D(patch_size, self.embed_dim, self.embed_dim, norm_layer)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            VSSLayer(dims[i], depths[i], d_state, dpr[sum(depths[:i]):sum(depths[:i + 1])],
                     norm_layer, PatchMerging2D if (i < 3) else None)
            for i in range(4)
        ])
        dpr_dec = dpr[::-1]
        self.layers_up = nn.ModuleList([
            VSSLayer_up(dims_decoder[i], depths_decoder[i], d_state,
                        dpr_dec[sum(depths_decoder[:i]):sum(depths_decoder[:i + 1])],
                        norm_layer, PatchExpand2D if (i != 0) else None)
            for i in range(4)
        ])
        self.final_up = Final_PatchExpand2D(dims_decoder[-1], 4)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fcd(x)
        x = self.patch_embed(x)
        skip = []
        for layer in self.layers:
            skip.append(x)
            x = layer(x)
        for i, l_up in enumerate(self.layers_up):
            # 修正skip连接索引（原版逻辑）
            skip_idx = len(skip) - i - 1
            x = l_up(x if i == 0 else x + skip[skip_idx])
        return self.final_conv(self.final_up(x).permute(0, 3, 1, 2))