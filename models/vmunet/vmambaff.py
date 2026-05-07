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

        # 2. 自适应对比度拉伸系数 (删除了破坏物理意义的 1x1 Conv，Gamma 仅作用于 1 个 L 通道)
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, 1))

        # 3. 原生的 Patch Embedding Projection
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(3, self.C, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(self.C)
        else:
            self.norm = None

    def clean_phase(self, x):
        """阶段 A: 物理清洗层 (先物理超度毛发)"""
        B, C, H, W = x.shape

        # 【全局图像 Min-Max 归一化】保护色调一致性
        x_min = x.view(B, -1).min(dim=-1)[0].view(B, 1, 1, 1)
        x_max = x.view(B, -1).max(dim=-1)[0].view(B, 1, 1, 1)
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)

        # ---------------- 找毛发位置 (在单通道灰度图上进行，最准) ----------------
        x_gray = x_norm.mean(dim=1, keepdim=True)
        dilation_gray = F.max_pool2d(x_gray, kernel_size=self.morph_kernel, stride=1, padding=self.padding)
        erosion_gray = -F.max_pool2d(-dilation_gray, kernel_size=self.morph_kernel, stride=1, padding=self.padding)
        hair_mask = torch.clamp(torch.abs(x_gray - erosion_gray) * 10.0, 0.0, 1.0)

        # ---------------- 准备回填皮肤 (在三通道RGB图上进行，保色) ----------------
        # 【最终修复】：对彩色图直接进行闭运算，得到色彩连续的平滑无毛皮肤底色
        dilation_rgb = F.max_pool2d(x_norm, kernel_size=self.morph_kernel, stride=1, padding=self.padding)
        erosion_rgb = -F.max_pool2d(-dilation_rgb, kernel_size=self.morph_kernel, stride=1, padding=self.padding)

        # 物理彩色回填！
        x_cleaned = x_norm * (1 - hair_mask) + erosion_rgb * hair_mask

        # 硬件级浮点安全锁
        return x_cleaned.clamp(1e-5, 1.0)

    def enhance_phase(self, x_cleaned):
        """阶段 B: 安全增强层 (在纯净 LAB 空间进行解耦亮度增强)"""
        # 1. 转换到 LAB 空间 (Kornia输出：L在约[0,100], a和b在约[-128,127])
        x_lab = kornia.color.rgb_to_lab(x_cleaned)

        # 2. 通道解耦：分离明暗(L)与色彩(A,B)
        L = x_lab[:, 0:1, :, :]
        A = x_lab[:, 1:2, :, :]
        B = x_lab[:, 2:3, :, :]

        # 3. 提取并归一化 L 通道 (限制在安全的浮点范围内供求幂计算)
        L_norm = L / 100.0
        L_norm = torch.clamp(L_norm, 1e-6, 1.0)

        # 4. 限制 Gamma 范围，防止求导时梯度爆炸
        gamma_safe = torch.clamp(torch.abs(self.gamma), min=0.5, max=3.0)

        # 5. 仅对亮度 L 进行非线性 Gamma 校正 (彻底保留原本病灶特征色彩)
        L_enhanced_norm = torch.pow(L_norm, gamma_safe)

        # 6. 还原 L 通道尺度
        L_enhanced = L_enhanced_norm * 100.0

        # 7. 重新拼接 LAB
        lab_enhanced = torch.cat([L_enhanced, A, B], dim=1)

        # 8. 转回 RGB 空间供后续 Patch Embedding 使用
        x_enhanced_rgb = kornia.color.lab_to_rgb(lab_enhanced)

        return x_enhanced_rgb.clamp(1e-5, 1.0)

    def forward(self, x):
        x_cleaned = self.clean_phase(x)
        x_enhanced = self.enhance_phase(x_cleaned)

        # 完美对接 VM-UNet 后续
        x_patched = self.proj(x_enhanced).permute(0, 2, 3, 1).contiguous()
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
        self.spiral_alpha = nn.Parameter(torch.full((1,), 0.1))
        # =========================================================================

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

        y_spiral_restored = out_y_1[:, 0][:, :, inv_idx_spiral]

        # 返回 5 个独立的张量
        return out_y_4[:, 0], inv_y[:, 0], wh_y, invwh_y, y_spiral_restored

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
                x = layer_up(x + skip_list[-inx])

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