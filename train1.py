import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
# 导入修复后的模型类
from models.vmunet.vmunet1 import VMUNet1, FocalDiceLoss
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def get_optimizer_with_lr_param(model, config):
    """分阶段学习率：为不同模块设置差异化学习率"""
    # 分组设置学习率：缺失权重的模块用高学习率，其余用基础学习率
    param_groups = [
        # 组1：PatchEmbed/SS2D（缺失预训练，高学习率）
        {
            "params": [p for n, p in model.named_parameters() if 'patch_embed.proj' in n or 'A_logs' in n or 'Ds' in n],
            "lr": config.lr * 10,  # 比基础学习率高10倍
            "weight_decay": config.weight_decay * 0.1  # 低权重衰减，快速收敛
        },
        # 组2：FCD模块（定制化模块，中学习率）
        {
            "params": [p for n, p in model.named_parameters() if 'fcd' in n],
            "lr": config.lr * 2,
            "weight_decay": config.weight_decay
        },
        # 组3：其余模块（有预训练，基础学习率）
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in ['patch_embed.proj', 'A_logs', 'Ds', 'fcd'])],
            "lr": config.lr,
            "weight_decay": config.weight_decay
        }
    ]
    if config.opt == 'AdamW':
        return torch.optim.AdamW(param_groups, betas=config.betas, eps=config.eps, amsgrad=config.amsgrad)
    else:
        return torch.optim.SGD(param_groups, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs, exist_ok=True)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
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

    print('#----------Prepareing Model (FCD-VMUNet1)----------#')
    model_cfg = config.model_config
    # 核心修改：使用修复后的 VMUNet1
    if config.network == 'vmunet':
        model = VMUNet1(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg.get('load_ckpt_path', None)
        )

        # 调用修复后的权重加载逻辑
        if model.load_ckpt_path is not None:
            model.load_from()

    else:
        raise Exception('network in not right!')

    model = model.cuda()

    # 针对 256x256 分辨率计算参数量和 FLOPs
    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = FocalDiceLoss(wf=0.3, wd=0.7)  # 优化后的损失函数
    optimizer = get_optimizer_with_lr_param(model, config)  # 分阶段学习率优化器
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()  # 混合精度训练

    print('#----------Set other params----------#')
    min_loss = 999
    best_dice = 0.0  # 基于Dice早停
    start_epoch = 1
    min_epoch = 1
    no_improve_epoch = 0  # 无改进轮数计数

    # 测试模式逻辑保持不变
    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        config.work_dir = config.img_save_path
        os.makedirs(config.work_dir + 'outputs/', exist_ok=True)
        loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        return

    # 断点续训逻辑保持不变
    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        best_dice = checkpoint.get('best_dice', 0.0)

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}, best_dice: {best_dice:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training with FCD Strategy----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        # 训练一轮（混合精度+梯度裁剪）
        model.train()
        epoch_loss = 0.0
        for step, (img, mask) in enumerate(train_loader):
            img, mask = img.cuda(), mask.cuda()
            with autocast():  # 混合精度训练
                pred, _ = model(img)
                loss = criterion(pred, mask)

            # 梯度裁剪 + 优化
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)  # 梯度裁剪
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            if step % config.print_interval == 0:
                logger.info(f'Epoch [{epoch}/{config.epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 学习率调度
        scheduler.step()

        # 验证阶段（计算Dice用于早停）
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.cuda(), mask.cuda()
                pred, _ = model(img)
                loss = criterion(pred, mask)
                val_loss += loss.item()
                # 计算Dice系数
                pred_bin = (pred > config.threshold).float()
                intersection = (pred_bin * mask).sum()
                union = pred_bin.sum() + mask.sum() + 1e-6
                dice = (2. * intersection + 1e-6) / union
                val_dice += dice.item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        logger.info(f'Epoch [{epoch}/{config.epochs}], Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')

        # 保存最优模型（基于Dice）
        if val_dice > best_dice:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            best_dice = val_dice
            min_loss = val_loss
            min_epoch = epoch
            no_improve_epoch = 0  # 重置无改进计数
            logger.info(f'Best model updated at epoch {epoch} with Dice {val_dice:.4f}')
        else:
            no_improve_epoch += 1

        # 保存最新模型
        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'best_dice': best_dice,
                'min_epoch': min_epoch,
                'loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

        # 早停逻辑（防止过拟合/震荡）
        if no_improve_epoch >= config.early_stop_patience:
            logger.info(f'Early stopping triggered: Dice no improvement for {config.early_stop_patience} epochs')
            break

    # 最终测试阶段：加载最优权重
    best_ckpt_path = os.path.join(checkpoint_dir, 'best.pth')
    if os.path.exists(best_ckpt_path):
        print('#----------Final Testing on Best Weights----------#')
        best_weight = torch.load(best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        # 重命名最优模型（标注epoch和Dice）
        new_best_name = f'best-epoch{min_epoch}-dice{best_dice:.4f}.pth'
        os.rename(best_ckpt_path, os.path.join(checkpoint_dir, new_best_name))
        logger.info(f'Best model saved as: {new_best_name}')


if __name__ == '__main__':
    config = setting_config
    main(config)