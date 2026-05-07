"""
diagnose_dt.py — 诊断 SS2D 各层 ∆t 在 hard sample 上的空间分布

目的：定位 hard sample 漏检的根本原因。每张图会输出：
  1. 每个 SS2D 模块（4+1 路径）的 ∆t 空间热力图
  2. GT 区域 vs 漏检区域 的 ∆t 平均值对比表
  3. 哪些层"看到"了漏检区，哪些层"忽略"了

用法：
  1. 修改下面的 BEST_CKPT_PATH 和 HARD_SAMPLE_PATHS
  2. python diagnose_dt.py
  3. 查看 ./dt_diagnosis/ 目录下的图

注意：
  - HARD_SAMPLE_PATHS 必须有对应的 GT mask（用于框定漏检区）
  - 如果你只想看模型预测和 ∆t 的关系，不传 mask 也行，但分析会粗糙
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.vmunet.vmunetff import VMUNet
from configs.config_setting import setting_config


# ============== 用户配置 ==============
BEST_CKPT_PATH = '/root/root/VM-UNet/results/vmunet_isic18_Sunday_03_May_2026_20h_44m_16s/checkpoints/early_epochs/epoch_012_miou0.8075.pth'

# Hard sample 列表：(原图路径, GT mask 路径)
# 如果某张图没有 GT，把 mask 路径写 None
HARD_SAMPLE_PATHS = [
    ('/root/root/VM-UNet/inputs/img_1.png', None),
    ('/root/root/VM-UNet/inputs/img_2.png', None),
    ('/root/root/VM-UNet/inputs/3.png', None),
]

OUT_DIR = './dt_diagnosis'
INPUT_SIZE = 256
DEVICE = 'cuda'
PATH_NAMES = ['lr_tb', 'tb_lr', 'rl_bt', 'bt_rl', 'spiral']  # 5 条路径名
# ====================================


# ============== Monkey-patch SS2D 让它记录 ∆t ==============
def make_recording_forward_core(original_func, ss2d_instance):
    """
    包装原 forward_core_decoupled，让它在每次 forward 时把 5 条路径的 ∆t 存到 ss2d_instance._dt_records
    """
    def wrapped(x):
        B, C, H, W = x.shape
        L = H * W

        # ===== 复制阶段 1（4 路径）的 ∆t 计算 =====
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs_4 = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl_4 = torch.einsum("b k d l, k c d -> b k c l", xs_4.view(B, 4, -1, L), ss2d_instance.x_proj_weight)
        dts_4, _, _ = torch.split(x_dbl_4, [ss2d_instance.dt_rank, ss2d_instance.d_state, ss2d_instance.d_state], dim=2)
        dts_4 = torch.einsum("b k r l, k d r -> b k d l", dts_4.view(B, 4, -1, L), ss2d_instance.dt_projs_weight)
        # 加 bias 和 softplus（和真实 selective_scan 内部一致）
        dts_4 = dts_4 + ss2d_instance.dt_projs_bias.view(4, -1, 1)  # broadcast
        dts_4 = F.softplus(dts_4)  # [B, 4, D, L]

        # ===== 复制阶段 2（spiral）的 ∆t 计算 =====
        device = x.device
        cache_key = (H, W, device)
        if cache_key not in ss2d_instance.spiral_cache:
            # 触发 cache 建立
            _ = original_func(x)  # 这会建好 cache
        idx_spiral, inv_idx_spiral = ss2d_instance.spiral_cache[cache_key]

        xs_1 = x.view(B, -1, L)[:, :, idx_spiral].unsqueeze(1)
        x_dbl_1 = torch.einsum("b k d l, c d -> b k c l", xs_1.view(B, 1, -1, L), ss2d_instance.x_proj_spiral)
        dts_1, _, _ = torch.split(x_dbl_1, [ss2d_instance.dt_rank, ss2d_instance.d_state, ss2d_instance.d_state], dim=2)
        dts_1 = torch.einsum("b k r l, d r -> b k d l", dts_1.view(B, 1, -1, L), ss2d_instance.dt_projs_weight_spiral)
        dts_1 = dts_1 + ss2d_instance.dt_projs_bias_spiral.view(1, -1, 1)
        dts_1 = F.softplus(dts_1)  # [B, 1, D, L]

        # ===== 把 spiral 路径的 ∆t 反扫描回原始空间顺序 =====
        # dts_1 是按 spiral 顺序排的，需要 inv_idx 还原
        dts_1_restored = dts_1[:, 0, :, inv_idx_spiral]  # [B, D, L]
        dts_1_restored = dts_1_restored.unsqueeze(1)  # [B, 1, D, L]

        # ===== 把 4 路径的 ∆t 也还原成原始空间顺序 =====
        # 路径 0: x.view(B,-1,L) - 原始顺序，不变
        # 路径 1: x.transpose(2,3) - HW 转置，需要 transpose 回来
        # 路径 2: 路径 0 的 flip  - flip 回来
        # 路径 3: 路径 1 的 flip  - flip 回来再 transpose 回来
        dts_4_restored = torch.zeros_like(dts_4)
        dts_4_restored[:, 0] = dts_4[:, 0]  # 已经是原始顺序
        # 路径 1：原本是 H,W 转置后展平，dt 也按这个顺序，需要 transpose 回来
        d_inner = dts_4.shape[2]
        dts_4_restored[:, 1] = dts_4[:, 1].view(B, d_inner, W, H).transpose(2, 3).contiguous().view(B, d_inner, L)
        # 路径 2：路径 0 flip 后扫描，flip 回来
        dts_4_restored[:, 2] = torch.flip(dts_4[:, 2], dims=[-1])
        # 路径 3：路径 1 flip 后扫描
        dts_4_restored[:, 3] = torch.flip(dts_4[:, 3], dims=[-1]).view(B, d_inner, W, H).transpose(2, 3).contiguous().view(B, d_inner, L)

        # 拼成 5 路径
        all_dts = torch.cat([dts_4_restored, dts_1_restored], dim=1)  # [B, 5, D, L]

        # 通道维平均，得到每个空间位置的 ∆t 重要性
        dt_per_position = all_dts.mean(dim=2)  # [B, 5, L]
        dt_per_position = dt_per_position.view(B, 5, H, W).cpu()

        # 存储
        ss2d_instance._dt_records = dt_per_position.detach()

        # 调用原 forward 返回真实结果
        return original_func(x)

    return wrapped


def attach_recorders(model):
    """遍历模型所有 SS2D 模块，给每个挂上 recording forward_core"""
    ss2d_modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'SS2D':
            original_func = module.forward_core_decoupled
            module.forward_core = make_recording_forward_core(original_func, module)
            module._dt_records = None
            module._layer_name = name
            ss2d_modules.append((name, module))
    return ss2d_modules


# ============== 主流程 ==============
def load_model_with_ckpt(ckpt_path):
    config = setting_config()
    model = VMUNet(
        num_classes=config.model_config['num_classes'],
        input_channels=config.model_config['input_channels'],
        depths=config.model_config['depths'],
        depths_decoder=config.model_config['depths_decoder'],
        drop_path_rate=config.model_config['drop_path_rate'],
        load_ckpt_path=None,
    )
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[Load] missing={len(missing)}, unexpected={len(unexpected)}')
    model = model.to(DEVICE).eval()
    return model, config


def preprocess_image(img_path, mask_path, config):
    """加载并预处理图像和 mask"""
    img = np.array(Image.open(img_path).convert('RGB'))
    if mask_path is not None and os.path.exists(mask_path):
        msk = np.expand_dims(np.array(Image.open(mask_path).convert('L')), axis=2) / 255.0
    else:
        msk = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.float32)

    img_t, msk_t = config.test_transformer((img, msk))
    img_tensor = img_t.unsqueeze(0).float().to(DEVICE)

    # resize 后的原图（用于显示）和 mask
    img_disp = np.array(Image.fromarray(img).resize((INPUT_SIZE, INPUT_SIZE)))
    if mask_path is not None and os.path.exists(mask_path):
        msk_disp = np.array(Image.fromarray((msk[:, :, 0] * 255).astype(np.uint8)).resize((INPUT_SIZE, INPUT_SIZE)))
        msk_disp = (msk_disp > 127).astype(np.float32)
    else:
        msk_disp = None

    return img_tensor, img_disp, msk_disp


def visualize_one_sample(model, ss2d_modules, img_tensor, img_disp, msk_disp, sample_name, out_dir):
    """forward 一次，可视化所有 SS2D 层的 ∆t"""
    # 清空所有记录
    for _, m in ss2d_modules:
        m._dt_records = None

    # forward
    with torch.no_grad():
        pred = model(img_tensor)
    pred_np = pred.cpu().numpy()[0, 0]

    # 收集所有有 ∆t 记录的层
    layers_with_dt = [(name, m) for name, m in ss2d_modules if m._dt_records is not None]
    n_layers = len(layers_with_dt)
    print(f'  Recorded ∆t from {n_layers} SS2D layers')

    # === 大总览图 ===
    # 每行：原图、GT、预测概率、5 条路径的 ∆t
    n_cols = 8  # img | gt | pred | 5 paths
    fig = plt.figure(figsize=(n_cols * 2.2, n_layers * 2.4))
    gs = gridspec.GridSpec(n_layers, n_cols, hspace=0.35, wspace=0.15)

    # 同时统计 GT 区 vs 漏检区的 ∆t 平均值
    stats_table = []  # 每行：[layer_name, path_name, dt_in_GT, dt_in_missed, ratio]

    for li, (lname, m) in enumerate(layers_with_dt):
        dt_records = m._dt_records[0]  # [5, H, W]
        H_dt, W_dt = dt_records.shape[1], dt_records.shape[2]

        # 把 GT 和 pred 缩放到 dt 的分辨率
        if msk_disp is not None:
            gt_at_dt = np.array(Image.fromarray((msk_disp * 255).astype(np.uint8))
                                .resize((W_dt, H_dt))) / 255.0
            gt_at_dt = (gt_at_dt > 0.5).astype(np.float32)
        else:
            gt_at_dt = None

        pred_at_dt = np.array(Image.fromarray((pred_np * 255).astype(np.uint8))
                              .resize((W_dt, H_dt))) / 255.0

        # 漏检区域 = GT 内 + 预测低
        if gt_at_dt is not None:
            missed_mask = gt_at_dt * (pred_at_dt < 0.3).astype(np.float32)
        else:
            missed_mask = None

        # === 第一列：缩短的 layer name ===
        short_name = lname.replace('vmunet.', '').replace('.self_attention', '')

        # === Col 0: 原图 with layer name ===
        ax = fig.add_subplot(gs[li, 0])
        ax.imshow(img_disp)
        ax.set_title(short_name, fontsize=8)
        ax.axis('off')

        # === Col 1: GT ===
        ax = fig.add_subplot(gs[li, 1])
        if msk_disp is not None:
            ax.imshow(msk_disp, cmap='gray', vmin=0, vmax=1)
        ax.set_title('GT' if li == 0 else '', fontsize=9)
        ax.axis('off')

        # === Col 2: 模型预测 ===
        ax = fig.add_subplot(gs[li, 2])
        ax.imshow(pred_np, cmap='jet', vmin=0, vmax=1)
        ax.set_title(f'Pred' if li == 0 else '', fontsize=9)
        ax.axis('off')

        # === Col 3-7: 5 条路径的 ∆t ===
        for pi in range(5):
            ax = fig.add_subplot(gs[li, 3 + pi])
            dt_map = dt_records[pi].numpy()
            # 用全图 max 归一化（避免每条路径自己 normalize 看不出差异）
            vmax = dt_map.max() if dt_map.max() > 0 else 1.0
            im = ax.imshow(dt_map, cmap='hot', vmin=0, vmax=vmax)
            ax.set_title(f'{PATH_NAMES[pi]} ({dt_map.max():.3f})' if li == 0 else f'{dt_map.max():.3f}',
                         fontsize=8)
            ax.axis('off')

            # 统计 GT 区和漏检区的 ∆t 平均值
            if gt_at_dt is not None and gt_at_dt.sum() > 5:
                dt_in_gt = (dt_map * gt_at_dt).sum() / (gt_at_dt.sum() + 1e-8)
                if missed_mask is not None and missed_mask.sum() > 5:
                    dt_in_missed = (dt_map * missed_mask).sum() / (missed_mask.sum() + 1e-8)
                    ratio = dt_in_missed / (dt_in_gt + 1e-8)
                    stats_table.append([short_name, PATH_NAMES[pi],
                                        dt_in_gt, dt_in_missed, ratio])

    plt.suptitle(f'∆t Diagnosis: {sample_name}', fontsize=12)
    save_path = os.path.join(out_dir, f'{sample_name}_dt_overview.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=80)
    plt.close()
    print(f'  Saved: {save_path}')

    # === 保存统计表 ===
    if len(stats_table) > 0:
        with open(os.path.join(out_dir, f'{sample_name}_stats.txt'), 'w') as f:
            f.write(f'∆t statistics for {sample_name}\n')
            f.write('=' * 90 + '\n')
            f.write(f'{"Layer":<35} {"Path":<8} {"dt_in_GT":>12} {"dt_in_missed":>14} {"ratio":>8}\n')
            f.write('-' * 90 + '\n')
            f.write('Notes:\n')
            f.write('  dt_in_GT     : average ∆t over the entire GT region\n')
            f.write('  dt_in_missed : average ∆t over the GT region that pred < 0.3 (the missed area)\n')
            f.write('  ratio        : dt_in_missed / dt_in_GT\n')
            f.write('               > 1 → model "sees" the missed area as much/more important than GT avg\n')
            f.write('               < 1 → model "ignores" the missed area relative to GT avg\n')
            f.write('               near 0 → model completely fails to attend to the missed area at this layer/path\n')
            f.write('=' * 90 + '\n')
            for row in stats_table:
                f.write(f'{row[0]:<35} {row[1]:<8} {row[2]:>12.5f} {row[3]:>14.5f} {row[4]:>8.3f}\n')
        print(f'  Saved stats table')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print('Loading model...')
    model, config = load_model_with_ckpt(BEST_CKPT_PATH)

    print('Attaching ∆t recorders...')
    ss2d_modules = attach_recorders(model)
    print(f'  Attached to {len(ss2d_modules)} SS2D modules')

    for img_path, mask_path in HARD_SAMPLE_PATHS:
        if not os.path.exists(img_path):
            print(f'[SKIP] {img_path} not found')
            continue
        sample_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f'\nProcessing: {sample_name}')

        img_tensor, img_disp, msk_disp = preprocess_image(img_path, mask_path, config)
        visualize_one_sample(model, ss2d_modules, img_tensor, img_disp, msk_disp,
                             sample_name, OUT_DIR)

    print('\nDone. Check ./dt_diagnosis/ for results.')


if __name__ == '__main__':
    main()