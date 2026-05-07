"""
放到项目根目录：finetuneff.py
跑法：python finetuneff.py
"""

import torch
from torch.utils.data import DataLoader
# ============== 修改点 1：彻底抛弃 7N / 10N，使用干净的原始数据流 ==============
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunetff import VMUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting_finetune import setting_config_finetune as setting_config

import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def freeze_for_direct_decoder_tuning(model):
    """
    抛弃 LoRA，直接全参解冻 Decoder 的 out_proj 和 final_conv。
    其余全部冻结（保护 Encoder 的特征提取能力）。
    """
    # 1. 先把全网五花大绑（全部冻结）
    for p in model.parameters():
        p.requires_grad = False

    trainable_count = 0
    # 2. 精准解冻目标层
    for name, p in model.named_parameters():
        # 目标 1：解冻 Decoder (layers_up) 中的 out_proj
        if 'layers_up' in name and 'out_proj' in name:
            p.requires_grad = True
            trainable_count += 1

        # 目标 2：解冻最终输出卷积 final_conv
        elif 'final_conv' in name:
            p.requires_grad = True
            trainable_count += 1

        # [强烈建议] 目标 3：顺手解冻 Decoder 的 LayerNorm
        # Norm 层参数极少，但对特征分布的拉扯能力极强，是抗毛发的关键
        elif ('layers_up' in name or 'final_up' in name or 'out_norm' in name) and 'norm' in name:
            p.requires_grad = True
            trainable_count += 1

    return trainable_count


def track_hard_samples(model, epoch, config, device):
    """
    极简版：单次推理，生成 3 面板对比图 (原图, 概率图, 叠加图)
    保留自 trainff.py，用于监控微调过程中困难样本（浅色/毛发）的概率响应变化。
    """
    model.eval()

    input_dir = getattr(config, 'hard_samples_input_dir', './inputs/')
    if not os.path.exists(input_dir):
        return

    hard_sample_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not hard_sample_paths:
        return

    tracking_dir = os.path.join(config.work_dir, 'hard_samples_tracking')
    os.makedirs(tracking_dir, exist_ok=True)

    with torch.no_grad():
        for img_path in hard_sample_paths:
            img_name = os.path.basename(img_path)
            original_img = Image.open(img_path).convert('RGB')
            original_img = original_img.resize((config.input_size_w, config.input_size_h))
            img_np = np.array(original_img)
            dummy_mask = np.zeros((config.input_size_h, config.input_size_w, 1), dtype=np.float32)
            img_tensor, _ = config.test_transformer((img_np, dummy_mask))
            img_tensor = img_tensor.unsqueeze(0).float().to(device)

            out = model(img_tensor)
            out = out[0] if isinstance(out, tuple) else out
            prob_max = out.max().item()
            prob_mean = out.mean().item()

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img_np)
            axes[0].set_title("Original")
            axes[0].axis('off')

            prob_f = out.squeeze().cpu().numpy()
            axes[1].imshow(prob_f, cmap='jet')
            axes[1].set_title(f"Ep{epoch} max={prob_max:.2f} mean={prob_mean:.3f}")
            axes[1].axis('off')

            axes[2].imshow(img_np)
            axes[2].imshow(prob_f, cmap='jet', alpha=0.5)
            axes[2].set_title("Overlay")
            axes[2].axis('off')

            save_path = os.path.join(tracking_dir, f"epoch_{epoch:03d}_{img_name}")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


def load_best_checkpoint_to_inner(model: VMUNet, ckpt_path: str, logger):
    """
    将 best.pth 加载到 model.vmunet（VSSM 实例）。
    best.pth 是用 model.state_dict() 直接保存的 VMUNet 全参数 dict。
    """
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']

    # VMUNet 内部是 self.vmunet = VSSM(...)，所以 key 形如 'vmunet.layers.0...'
    # 直接用 strict=False 加载即可
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"[Load] missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        logger.info(f"  first missing: {missing[:5]}")
    if len(unexpected) > 0:
        logger.info(f"  first unexpected: {unexpected[:5]}")
    return model


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs): os.makedirs(outputs)

    global logger
    logger = get_logger('finetune', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============== 加载纯净版数据集 ==============
    print('#----------Preparing Clean Dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    logger.info(f"Clean train set length: {len(train_dataset)} (No 7N/10N residual operations)")

    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)

    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Preparing Model + Loading Epoch 8 Checkpoint----------#')
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=None,
    )
    model = load_best_checkpoint_to_inner(model, config.best_ckpt_path, logger)
    model = model.cuda()

    # ============== 核心修改点：直接全参解冻 Decoder ==============
    print('#---------- Freezing Encoder & Tuning Decoder directly ----------#')
    trainable_count = freeze_for_direct_decoder_tuning(model)

    logger.info(
        f"[Direct Tuning] Unfrozen {trainable_count} parameter tensors in Decoder (out_proj, final_conv, norms).")
    logger.info(f"Encoder is fully frozen and protected.")

    # 打印看看到底解冻了哪些层，确认没误伤
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.info(f"  Trainable: {name}")

    cal_params_flops(model, 256, logger)

    print('#----------Preparing loss, opt, sch----------#')
    criterion = config.criterion

    # Optimizer 只接收 requires_grad=True 的参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"[Optim] feeding {sum(p.numel() for p in trainable_params):,} params to optimizer")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        amsgrad=config.amsgrad,
    )
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    max_miou = 0.0
    start_epoch = 1
    best_epoch = 1

    logger.info("#---------- Sanity val before finetune ----------#")
    val_loss_before, miou_before = val_one_epoch(
        val_loader, model, criterion, 0, logger, config
    )
    logger.info(f"[Before finetune] val_loss={val_loss_before:.4f}, mIoU={miou_before:.4f}")

    try:
        track_hard_samples(model, 0, config, device)
        logger.info("[Tracking] baseline (epoch 0) snapshot saved")
    except Exception as e:
        logger.warning(f"困难样本追踪失败 (baseline): {e}")

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        max_miou, best_epoch = checkpoint.get('max_miou', 0.0), checkpoint.get('best_epoch', 1)
        logger.info(f'resume: epoch={saved_epoch}, max_miou={max_miou:.4f}')

    step = 0
    print('#----------Finetuning----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader, model, criterion, optimizer, scheduler,
            epoch, step, logger, config, writer
        )

        val_loss, current_miou = val_one_epoch(
            val_loader, model, criterion, epoch, logger, config
        )

        if current_miou > max_miou:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            max_miou = current_miou
            best_epoch = epoch
            logger.info(f'New best mIoU: {max_miou:.4f} at epoch {epoch}')

        torch.save(
            {
                'epoch': epoch,
                'max_miou': max_miou,
                'best_epoch': best_epoch,
                'loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

        try:
            track_hard_samples(model, epoch, config, device)
        except Exception as e:
            logger.warning(f"困难样本追踪失败: {e}")

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Final Test----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'),
                                 map_location=torch.device('cpu'))
        model.load_state_dict(best_weight, strict=False)
        loss = test_one_epoch(val_loader, model, criterion, logger, config)
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-miou{max_miou:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config()
    main(config)