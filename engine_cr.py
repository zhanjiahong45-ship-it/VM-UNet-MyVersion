"""
train_one_epoch_cr
====================
CR 训练版的 train_one_epoch 函数。

放在 engine.py 的旁边、或者直接 paste 进 engine.py。
不替换原有的 train_one_epoch（保留作为 baseline 对照用），新增一个名字。

Usage in trainff.py:
    from engine_cr import train_one_epoch_cr
    step = train_one_epoch_cr(
        train_loader, model, criterion, consistency_loss_fn,
        optimizer, scheduler, epoch, step, logger, config, writer,
        lambda_cons=0.5,    # consistency loss weight
    )

Note: 配套使用 NPY_datasets_CR (返回 (img_normal, img_aug, msk))
"""

import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler


# Lazy global scaler (与原 engine 行为对齐, 如果你原 engine 已有 scaler 也可以传进来)
_scaler = None


def _get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = GradScaler()
    return _scaler


def train_one_epoch_cr(
    train_loader,
    model,
    criterion,                 # BceDiceLoss 等
    consistency_loss_fn,       # ConsistencyLoss 实例
    optimizer,
    scheduler,
    epoch,
    step,
    logger,
    config,
    writer,
    lambda_cons: float = 0.5,
    log_lambda_warmup: bool = True,
):
    """
    CR 训练循环：
        L = L_seg(pred_normal, gt) + L_seg(pred_aug, gt) + lambda_cons * L_cons(pred_normal, pred_aug)

    Important:
        - DataLoader 必须返回 (img_normal, img_aug, msk) 三元组（用 NPY_datasets_CR）
        - lambda_cons 控制一致性损失权重
        - 前 N epoch 可以 warmup lambda_cons (这里默认线性 warmup 前 5 epoch)

    Returns:
        step (int): 全局 iteration step
    """
    model.train()
    loss_list = []
    seg_loss_list = []
    cons_loss_list = []

    # lambda warmup: 前 5 epoch 从 0 线性涨到 lambda_cons
    warmup_epochs = 5
    if epoch <= warmup_epochs:
        cur_lambda = lambda_cons * (epoch / warmup_epochs)
    else:
        cur_lambda = lambda_cons

    if log_lambda_warmup and epoch <= warmup_epochs + 1:
        logger.info(f'[CR] epoch {epoch}: lambda_cons={cur_lambda:.4f}')

    use_amp = getattr(config, 'amp', False)
    scaler = _get_scaler() if use_amp else None

    for iter_idx, batch in enumerate(train_loader):
        step += iter_idx

        # ---- unpack: (img_normal, img_aug, msk) ----
        img_normal, img_aug, msk = batch
        img_normal = img_normal.cuda(non_blocking=True).float()
        img_aug = img_aug.cuda(non_blocking=True).float()
        msk = msk.cuda(non_blocking=True).float()

        optimizer.zero_grad()

        if use_amp:
            # 1. Only run the forward pass inside autocast
            with autocast():
                pred_normal = model(img_normal)
                pred_aug = model(img_aug)
                pred_normal = pred_normal[0] if isinstance(pred_normal, tuple) else pred_normal
                pred_aug = pred_aug[0] if isinstance(pred_aug, tuple) else pred_aug

            # 2. Move loss calculations OUTSIDE autocast and cast to float32
            pred_normal = pred_normal.float()
            pred_aug = pred_aug.float()

            loss_seg = criterion(pred_normal, msk) + criterion(pred_aug, msk)
            loss_cons = consistency_loss_fn(pred_normal, pred_aug)
            loss = loss_seg + cur_lambda * loss_cons

            # 3. Backward pass remains the same
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_normal = model(img_normal)
            pred_aug = model(img_aug)
            pred_normal = pred_normal[0] if isinstance(pred_normal, tuple) else pred_normal
            pred_aug = pred_aug[0] if isinstance(pred_aug, tuple) else pred_aug

            loss_seg = criterion(pred_normal, msk) + criterion(pred_aug, msk)
            loss_cons = consistency_loss_fn(pred_normal, pred_aug)
            loss = loss_seg + cur_lambda * loss_cons

            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())
        seg_loss_list.append(loss_seg.item())
        cons_loss_list.append(loss_cons.item())

        # 日志
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter_idx % config.print_interval == 0:
            log_info = (
                f'train: epoch {epoch}, iter:{iter_idx}, '
                f'loss: {loss.item():.4f}, '
                f'seg: {loss_seg.item():.4f}, '
                f'cons: {loss_cons.item():.4f} (λ={cur_lambda:.3f}), '
                f'lr: {now_lr}'
            )
            logger.info(log_info)
            print(log_info)

            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/seg_loss', loss_seg.item(), step)
                writer.add_scalar('train/cons_loss', loss_cons.item(), step)
                writer.add_scalar('train/lambda_cons', cur_lambda, step)

    scheduler.step()

    # epoch 结束日志
    avg_total = np.mean(loss_list)
    avg_seg = np.mean(seg_loss_list)
    avg_cons = np.mean(cons_loss_list)
    logger.info(
        f'epoch {epoch} avg: total={avg_total:.4f}, seg={avg_seg:.4f}, cons={avg_cons:.4f}'
    )

    return step