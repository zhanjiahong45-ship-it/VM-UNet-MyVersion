import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import kornia

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L

    return flops


class Adaptive_SASF_v2(nn.Module):
    """
    Production-grade SASF with:
    1. Channel-split multi-dilation (B2): 67% FLOPs reduction
    2. Fused BN-ReLU before projection for better gradient flow
    3. Zero-init residual preserved
    """
    def __init__(self, dim, resolution):
        super().__init__()

        if resolution >= 32:
            dilations = (1, 3, 5)
        elif resolution == 16:
            dilations = (1, 2, 4)
        else:
            dilations = (1, 2, 3)

        # Channel-split sizes (handle non-divisible-by-3)
        g = dim // 3
        self.group_sizes = [g, g, dim - 2 * g]

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(gs, gs, kernel_size=3,
                          padding=d, dilation=d, groups=gs, bias=False),
                nn.BatchNorm2d(gs),
                nn.ReLU(inplace=True)
            )
            for gs, d in zip(self.group_sizes, dilations)
        ])

        # Projection with zero-init
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x_2d):
        groups = torch.split(x_2d, self.group_sizes, dim=1)
        outs = [conv(g) for conv, g in zip(self.dwconvs, groups)]
        out = self.proj(torch.cat(outs, dim=1))
        return x_2d + out
# ==========================================
# 👑 创新模块：自适应残差语义桥接路径 (ResPath)
# 解决 Encoder(浅层) 与 Decoder(深层) 的语义鸿沟
# ==========================================
# ==========================================
# 👑 创新模块：轻量化残差语义桥接路径 (Lightweight ResPath)
# 采用深度可分离卷积，完美解决高维通道下的参数暴增问题
# ==========================================
class LightConv2d_batchnorm(torch.nn.Module):
    def __init__(self, num_in_filters, num_out_filters, kernel_size, stride=(1, 1), activation='relu'):
        super().__init__()
        self.activation = activation

        # 判断如果是 1x1 卷积 (Shortcut使用)，直接用常规卷积
        if kernel_size == 1 or kernel_size == (1, 1):
            self.conv = torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters,
                                        kernel_size=1, stride=stride)
        else:
            # 如果是 3x3 卷积 (主干使用)，采用深度可分离卷积 (Depthwise Separable Conv)
            padding = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, tuple) else kernel_size // 2
            self.conv = torch.nn.Sequential(
                # 第一步：Depthwise 空间提取 (groups=in_channels)，参数量极低
                torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_in_filters,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=num_in_filters),
                # 第二步：Pointwise 跨通道融合
                torch.nn.Conv2d(in_channels=num_in_filters, out_channels=num_out_filters, kernel_size=1)
            )

        self.batchnorm = torch.nn.BatchNorm2d(num_out_filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activation == 'relu':
            return torch.nn.functional.relu(x)
        return x


# ==========================================
# 👑 创新模块：注意力引导的轻量化残差路径 (Attention-Guided Lightweight ResPath)
# 解决后期网络过于保守而过滤浅色病灶的痛点
# ==========================================
class ResPath(torch.nn.Module):
    def __init__(self, num_in_filters, num_out_filters, respath_length):
        super().__init__()
        self.respath_length = respath_length
        self.shortcuts = torch.nn.ModuleList([])
        self.convs = torch.nn.ModuleList([])
        self.bns = torch.nn.ModuleList([])

        for i in range(self.respath_length):
            in_channels = num_in_filters if i == 0 else num_out_filters
            self.shortcuts.append(
                LightConv2d_batchnorm(in_channels, num_out_filters, kernel_size=(1, 1), activation='None'))
            self.convs.append(
                LightConv2d_batchnorm(in_channels, num_out_filters, kernel_size=(3, 3), activation='relu'))
            self.bns.append(torch.nn.BatchNorm2d(num_out_filters))

        # 🌟【新增】：轻量级空间注意力 (Spatial Attention)
        # 仅增加几十个参数，极度廉价却能赋予网络空间筛选能力
        self.spatial_attention = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()

        for i in range(self.respath_length):
            shortcut = self.shortcuts[i](x)

            x_main = self.convs[i](x)
            x_main = self.bns[i](x_main)
            x_main = torch.nn.functional.relu(x_main)

            x = x_main + shortcut
            x = self.bns[i](x)
            x = torch.nn.functional.relu(x)

        # 🌟【新增】：在送出浅层特征之前，进行空间注意力筛选
        # 分别提取通道的平均池化和最大池化特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接后计算注意力权重 (0~1 之间)
        attention_map = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))

        # 将不重要的毛发区域压制，突出真正的浅色病灶区域！
        x = x * (1.0 + attention_map)

        x = x.permute(0, 2, 3, 1).contiguous()
        return x


# ==========================================


class FCD_Module(nn.Module):
    r"""
    FCD-Module v2: Parallel Clean/Enhance with Learnable Routing
    - clean_phase: morphological hair removal (unchanged)
    - enhance_phase: LAB-space gamma with LEARNABLE gamma parameter
    - routing: per-pixel soft blend between cleaned and enhanced streams
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        self.C = embed_dim
        self.patch_size = patch_size

        # 1. Morphological operator kernel
        self.morph_kernel = 5
        self.padding = self.morph_kernel // 2

        # 2. 【Fix 1】LEARNABLE gamma — initialized to 0.7 (< 1 brightens)
        #    nn.Parameter, NOT register_buffer
        self.gamma = nn.Parameter(torch.tensor([0.7]).view(1, 1, 1, 1))

        # 3. 【Fix 5】Per-pixel routing network
        #    Input: 6 channels (3 from cleaned + 3 from enhanced)
        #    Output: 1 channel sigmoid mask (0 = use cleaned, 1 = use enhanced)
        self.blend_net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Initialize blend_net final layer bias to 0 → sigmoid(0)=0.5
        # so both streams contribute equally at the start
        nn.init.constant_(self.blend_net[3].bias, 0.0)

        # 4. Patch Embedding Projection (unchanged)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(3, self.C, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(self.C)
        else:
            self.norm = None

    def clean_phase(self, x):
        """Phase A: Morphological hair removal (fully preserved from your version)"""
        B, C, H, W = x.shape

        x_min = x.view(B, -1).min(dim=-1)[0].view(B, 1, 1, 1)
        x_max = x.view(B, -1).max(dim=-1)[0].view(B, 1, 1, 1)
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)

        x_gray = x_norm.mean(dim=1, keepdim=True)
        dilation_gray = F.max_pool2d(x_gray, kernel_size=self.morph_kernel,
                                     stride=1, padding=self.padding)
        erosion_gray = -F.max_pool2d(-dilation_gray, kernel_size=self.morph_kernel,
                                     stride=1, padding=self.padding)

        hair_mask = torch.clamp(torch.abs(x_gray - erosion_gray) * 10.0, 0.0, 1.0)
        hair_mask = F.max_pool2d(hair_mask, kernel_size=3, stride=1, padding=1)

        dilation_rgb = F.max_pool2d(x_norm, kernel_size=self.morph_kernel,
                                    stride=1, padding=self.padding)
        erosion_rgb = -F.max_pool2d(-dilation_rgb, kernel_size=self.morph_kernel,
                                    stride=1, padding=self.padding)
        smooth_erosion_rgb = F.avg_pool2d(erosion_rgb, kernel_size=3, stride=1, padding=1)

        x_cleaned_final = x_norm * (1 - hair_mask) + smooth_erosion_rgb * hair_mask
        return x_cleaned_final.clamp(1e-5, 1.0)

    def enhance_phase(self, x_input):
        """Phase B: LAB gamma correction — now from ORIGINAL input, not cleaned"""
        x_lab = kornia.color.rgb_to_lab(x_input)

        L = x_lab[:, 0:1, :, :]
        A = x_lab[:, 1:2, :, :]
        B_ch = x_lab[:, 2:3, :, :]  # renamed to avoid shadowing

        L_norm = torch.clamp(L / 100.0, 1e-6, 1.0)

        # 【Fix 1】gamma is now a learnable Parameter
        gamma_safe = torch.clamp(self.gamma.abs(), min=0.3, max=2.0)
        L_enhanced_norm = torch.pow(L_norm, gamma_safe)

        L_enhanced = L_enhanced_norm * 100.0
        lab_enhanced = torch.cat([L_enhanced, A, B_ch], dim=1)
        x_enhanced_rgb = kornia.color.lab_to_rgb(lab_enhanced)
        return x_enhanced_rgb.clamp(1e-5, 1.0)

    def forward(self, x):
        # 【Fix 5 — Critical Change】: Parallel, not sequential
        # Normalize input to [0,1] for both branches
        B, C, H, W = x.shape
        x_min = x.view(B, -1).min(dim=-1)[0].view(B, 1, 1, 1)
        x_max = x.view(B, -1).max(dim=-1)[0].view(B, 1, 1, 1)
        x_01 = (x - x_min) / (x_max - x_min + 1e-6)

        x_cleaned = self.clean_phase(x)        # hair removal stream
        x_enhanced = self.enhance_phase(x_01)   # contrast boost stream (from original)

        # Per-pixel routing: network sees both candidates and decides blend
        blend_input = torch.cat([x_cleaned, x_enhanced], dim=1)  # (B, 6, H, W)
        alpha = self.blend_net(blend_input)  # (B, 1, H, W), range [0, 1]

        # alpha≈0 → use cleaned (good for hairy regions)
        # alpha≈1 → use enhanced (good for light lesions)
        x_fused = (1 - alpha) * x_cleaned + alpha * x_enhanced

        # Standard patch embedding
        x_patched = self.proj(x_fused).permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x_patched = self.norm(x_patched)
        return x_patched


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer. """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # ================= 预训练的 K=4 部分 (必须严格保持命名与维度) =================
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # shape: [4, N, inner]
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0))  # shape: [4, inner, rank]
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # shape: [4, inner]
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        # ================= 新增的 第 5 条螺旋路径专属参数 (K=1) =================
        self.x_proj_spiral = nn.Parameter(
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs).weight)

        dt_proj_spiral = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                      **factory_kwargs)
        self.dt_projs_weight_spiral = nn.Parameter(dt_proj_spiral.weight)
        self.dt_projs_bias_spiral = nn.Parameter(dt_proj_spiral.bias)

        self.A_logs_spiral = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds_spiral = self.D_init(self.d_inner, copies=1, merge=True)

        # 【核心创新】：0初始化门控，防止新增路径破坏预训练平衡
        self.spiral_alpha = nn.Parameter(torch.zeros(1))

        # Compute resolution estimate FIRST
        if self.d_model <= 96:
            res_estimate = 64
        elif self.d_model <= 192:
            res_estimate = 32
        elif self.d_model <= 384:
            res_estimate = 16
        else:
            res_estimate = 8

        # Then conditionally create SASF
        self.use_sasf = (192 <= self.d_inner <= 768)
        if self.use_sasf:
            self.sasf_spiral = Adaptive_SASF_v2(dim=self.d_inner, resolution=res_estimate)

        self.forward_core = self.forward_core_decoupled

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.spiral_cache = {}

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core_decoupled(self, x: torch.Tensor):
        """
        分为两阶段分别进行 Mamba 扫描，彻底避免维度冲突
        """
        # 为了兼容不同的 mamba_ssm 版本
        try:
            self.selective_scan = selective_scan_fn
            v1_flag = False
        except:
            self.selective_scan = selective_scan_fn_v1
            v1_flag = True

        B, C, H, W = x.shape
        L = H * W

        # ================= 阶段 1: 处理预训练的 4 条基准路径 =================
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs_4 = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, 4, d, l)

        x_dbl_4 = torch.einsum("b k d l, k c d -> b k c l", xs_4.view(B, 4, -1, L), self.x_proj_weight)
        dts_4, Bs_4, Cs_4 = torch.split(x_dbl_4, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts_4 = torch.einsum("b k r l, k d r -> b k d l", dts_4.view(B, 4, -1, L), self.dt_projs_weight)

        xs_4 = xs_4.float().view(B, -1, L)
        dts_4 = dts_4.contiguous().float().view(B, -1, L)
        Bs_4 = Bs_4.float().view(B, 4, -1, L)
        Cs_4 = Cs_4.float().view(B, 4, -1, L)
        Ds_4 = self.Ds.float().view(-1)
        As_4 = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias_4 = self.dt_projs_bias.float().view(-1)

        if not v1_flag:
            out_y_4 = self.selective_scan(
                xs_4, dts_4, As_4, Bs_4, Cs_4, Ds_4, z=None,
                delta_bias=dt_projs_bias_4, delta_softplus=True, return_last_state=False,
            ).view(B, 4, -1, L)
        else:
            out_y_4 = self.selective_scan(
                xs_4, dts_4, As_4, Bs_4, Cs_4, Ds_4,
                delta_bias=dt_projs_bias_4, delta_softplus=True,
            ).view(B, 4, -1, L)

        assert out_y_4.dtype == torch.float

        # ================= 阶段 2: 处理新增的 1 条螺旋扩张路径 =================
        device = x.device
        cache_key = (H, W, device)

        if cache_key not in self.spiral_cache:
            idx = []
            top, bottom, left, right = 0, H - 1, 0, W - 1
            matrix = torch.arange(L, device=device).view(H, W)
            # 生成边缘向中心顺时针螺旋
            while top <= bottom and left <= right:
                idx.append(matrix[top, left:right + 1])
                top += 1
                if top <= bottom:
                    idx.append(matrix[top:bottom + 1, right])
                    right -= 1
                if top <= bottom and left <= right:
                    idx.append(matrix[bottom, left:right + 1].flip(0))
                    bottom -= 1
                if top <= bottom and left <= right:
                    idx.append(matrix[top:bottom + 1, left].flip(0))
                    left += 1
            idx_cw_inward = torch.cat(idx)
            # 翻转：得到中心向边缘的逆时针扩张螺旋
            idx_ccw_outward = idx_cw_inward.flip(0)

            inv_idx = torch.empty_like(idx_ccw_outward)
            inv_idx[idx_ccw_outward] = torch.arange(L, device=device)
            self.spiral_cache[cache_key] = (idx_ccw_outward, inv_idx)

        idx_spiral, inv_idx_spiral = self.spiral_cache[cache_key]

        xs_1 = x.view(B, -1, L)[:, :, idx_spiral].unsqueeze(1)  # (b, 1, d, l)

        x_dbl_1 = torch.einsum("b k d l, c d -> b k c l", xs_1.view(B, 1, -1, L), self.x_proj_spiral)
        dts_1, Bs_1, Cs_1 = torch.split(x_dbl_1, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts_1 = torch.einsum("b k r l, d r -> b k d l", dts_1.view(B, 1, -1, L), self.dt_projs_weight_spiral)

        xs_1 = xs_1.float().view(B, -1, L)
        dts_1 = dts_1.contiguous().float().view(B, -1, L)
        Bs_1 = Bs_1.float().view(B, 1, -1, L)
        Cs_1 = Cs_1.float().view(B, 1, -1, L)
        Ds_1 = self.Ds_spiral.float().view(-1)
        As_1 = -torch.exp(self.A_logs_spiral.float()).view(-1, self.d_state)
        dt_projs_bias_1 = self.dt_projs_bias_spiral.float().view(-1)

        if not v1_flag:
            out_y_1 = self.selective_scan(
                xs_1, dts_1, As_1, Bs_1, Cs_1, Ds_1, z=None,
                delta_bias=dt_projs_bias_1, delta_softplus=True, return_last_state=False,
            ).view(B, 1, -1, L)
        else:
            out_y_1 = self.selective_scan(
                xs_1, dts_1, As_1, Bs_1, Cs_1, Ds_1,
                delta_bias=dt_projs_bias_1, delta_softplus=True,
            ).view(B, 1, -1, L)

        # ================= 阶段 3: 逆向映射还原回 2D 空间 =================
        inv_y = torch.flip(out_y_4[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y_4[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        # 1. 螺旋路径原本的一维还原 shape: (B, d_inner, L)
        y_spiral_restored = out_y_1[:, 0][:, :, inv_idx_spiral]

        # -----------------------------------------------------------
        # 👑 终极融合：将 1D 螺旋序列的空间错位，通过 Adaptive-SASF 进行二维接骨修复
        # -----------------------------------------------------------
        # 2. 将还原后的一维特征重新折叠为真实的 2D 图像格式 shape: (B, d_inner, H, W)
        y_spiral_2d = y_spiral_restored.view(B, -1, H, W)
        if self.use_sasf:
            y_spiral_sasf_2d = self.sasf_spiral(y_spiral_2d)
        else:
            y_spiral_sasf_2d = y_spiral_2d
        y_spiral_final = y_spiral_sasf_2d.view(B, -1, L)
        # -----------------------------------------------------------

        # 返回 5 个独立的张量 (第 5 个现已具备自适应多尺度 2D 结构感知能力)
        return out_y_4[:, 0], inv_y[:, 0], wh_y, invwh_y, y_spiral_final

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)

        y1, y2, y3, y4, y5 = self.forward_core(x)

        # =================【最关键融合】=================
        # 前 4 条路径由于继承了预训练权重，具备极强的全局感知力，直接相加。
        # 第 5 条螺旋路径起初随机，受门控 spiral_alpha(=0) 压制；
        # 随着训练，模型学习到它的扩张补全能力，alpha 将变大。
        y = y1 + y2 + y3 + y4 + (self.spiral_alpha * y5)
        # ==============================================

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96], d_state=16, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        # 挂载带有物理清洗和自适应增强机制的 FCD_Module
        self.patch_embed = FCD_Module(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                      norm_layer=norm_layer if patch_norm else None)

        self.ape = False
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)
        self.respaths = nn.ModuleList()
        # VM-UNet 的 forward_features_up 实际上只用到 3 个跳跃连接
        # 从深到浅分别是 8C, 4C, 2C 对应的特征层
        skip_dims = [dims[3], dims[2], dims[1]]
        # 动态深度控制：深层语义鸿沟小(用3个块)，浅层语义鸿沟大(用1个块)
        respath_lengths = [3, 2, 1]

        for i in range(3):
            self.respaths.append(ResPath(num_in_filters=skip_dims[i],
                                         num_out_filters=skip_dims[i],
                                         respath_length=respath_lengths[i]))

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                # 【新增】通过 ResPath 对浅层特征进行语义跃迁
                # respaths[0, 1, 2] 对应于 inx=1, 2, 3
                processed_skip = self.respaths[inx - 1](skip_list[-inx])

                # 语义对齐后的完美融合
                x = layer_up(x + processed_skip)

        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)

        return x