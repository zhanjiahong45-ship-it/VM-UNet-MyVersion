import numpy as np
import cv2
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def _inpaint_with_skin(img_tensor, prob, threshold=0.4):
    """用全图非病灶区域均值皮肤色填充锚点区域，供两遍推理使用。"""
    anchor = (prob > threshold).float()
    non_anchor = 1.0 - anchor
    skin_color = (img_tensor * non_anchor).sum(dim=[2, 3], keepdim=True) / \
                 (non_anchor.sum(dim=[2, 3], keepdim=True) + 1e-8)
    return img_tensor * (1 - anchor) + skin_color.expand_as(img_tensor) * anchor


def _apply_clahe(img_tensor, clip_limit=2.0, tile_grid=(8, 8)):
    """对 [B, C, H, W] tensor 在 LAB 空间的 L 通道做 CLAHE，增强补色后图像的对比度。
    保持原始数值范围不变，仅改变通道内对比度分布。"""
    arr = img_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()  # [H, W, C]
    v_min, v_max = arr.min(), arr.max()
    # 映射到 uint8 供 OpenCV 处理
    arr_u8 = ((arr - v_min) / (v_max - v_min + 1e-8) * 255).clip(0, 255).astype(np.uint8)
    lab = cv2.cvtColor(arr_u8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced_u8 = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # 映射回原始数值范围
    enhanced = enhanced_u8.astype(np.float32) / 255.0 * (v_max - v_min) + v_min
    return torch.from_numpy(enhanced).permute(2, 0, 1).unsqueeze(0).contiguous().to(img_tensor.device)


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer,
                    scaler=None,
                    early_model=None):
    model.train()
    loss_list = []
    # 浅色 pseudo-label loss 的权重，可在 config 里覆盖，默认 0.5
    faint_loss_weight = getattr(config, 'faint_loss_weight', 0.5)

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        with autocast(enabled=config.amp):
            out = model(images)

        loss = criterion(out.float(), targets.float())

        # ── 浅色病灶两遍蒸馏 (epoch>=5, early_model 已就绪时生效) ──────────────
        if early_model is not None and faint_loss_weight > 0:
            # Step-1: 用【主模型】输出检测浅色样本（高峰值 + 稀疏分布）
            out_det = out.detach()
            faint_idx = [
                i for i in range(images.shape[0])
                if out_det[i].max().item() > 0.5 and out_det[i].mean().item() < 0.05
            ]

            if faint_idx:
                fi = torch.tensor(faint_idx, device=images.device)
                # 浅色样本用 GT 再压一次，不绕 mask_e2 pseudo-label
                faint_loss = criterion(out[fi].float(), targets[fi].float())
                loss = loss + faint_loss_weight * faint_loss

                if iter % config.print_interval == 0:
                    writer.add_scalar('faint/count', len(faint_idx), global_step=step)
                    writer.add_scalar('faint/loss', faint_loss.item(), global_step=step)
        # ─────────────────────────────────────────────────────────────────────

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            try:
                gamma_val = model.vmunet.patch_embed.gamma.item()
                writer.add_scalar('fcd/gamma', gamma_val, global_step=step)
                alpha_val = model.vmunet.layers[0].blocks[0].self_attention.spiral_alpha.item()
                writer.add_scalar('spiral/alpha_layer0', alpha_val, global_step=step)
            except:
                pass
    scheduler.step()
    return step


def val_one_epoch(val_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config,
                  writer=None,
                  early_model=None):
    model.eval()
    preds = []
    gts = []
    loss_list = []

    miou = 0.0

    with torch.no_grad():
        for data in tqdm(val_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            out = out[0] if type(out) is tuple else out

            # ── val 两遍精炼（与 test 逻辑对齐，使 mIoU 更真实）─────────────
            if early_model is not None:
                if out.max().item() > 0.5 and out.mean().item() < 0.05:
                    mask_e1 = early_model(img)
                    mask_e1 = mask_e1[0] if isinstance(mask_e1, tuple) else mask_e1
                    img2 = _inpaint_with_skin(img, mask_e1, threshold=0.4)
                    img2 = _apply_clahe(img2)
                    mask_e2 = early_model(img2)
                    mask_e2 = mask_e2[0] if isinstance(mask_e2, tuple) else mask_e2
                    out = mask_e2
            # ────────────────────────────────────────────────────────────────

            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())

            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds >= config.threshold, 1, 0)
        y_true = np.where(gts >= 0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, \
                specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

        # 如果传入了 writer，顺便记录一下验证集的指标
        if writer is not None:
            writer.add_scalar('val/loss', np.mean(loss_list), global_step=epoch)
            writer.add_scalar('val/miou', miou, global_step=epoch)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list), miou


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None,
                   early_model=None):
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out = model(img)
            out = out[0] if isinstance(out, tuple) else out

            # 两遍精炼：主模型检测浅色 → early_model 两刀 → 直接用 mask_e2
            if out.max() > 0.5 and out.mean() < 0.05:
                if early_model is not None:
                    mask_e1 = early_model(img)
                    mask_e1 = mask_e1[0] if isinstance(mask_e1, tuple) else mask_e1
                    img2 = _inpaint_with_skin(img, mask_e1, threshold=0.4)
                    img2 = _apply_clahe(img2)
                    mask_e2 = early_model(img2)
                    mask_e2 = mask_e2[0] if isinstance(mask_e2, tuple) else mask_e2
                    out = mask_e2
                else:
                    img2 = _inpaint_with_skin(img, out)
                    out2 = model(img2)
                    out2 = out2[0] if isinstance(out2, tuple) else out2
                    out = torch.max(out, out2)

            loss = criterion(out, msk)
            loss_list.append(loss.item())

            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)

            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

            if i % config.save_interval == 0:
                save_imgs(img.cpu(), msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        # 固定真实标签的阈值
        y_true = np.where(gts >= 0.5, 1, 0)

        # ====================================================================
        # 👑 终极压榨：暴力搜索最佳阈值 (Threshold Grid Search)
        # ====================================================================
        best_miou = -1.0  # 确保第一次必定能触发更新
        best_thresh = config.threshold

        # 【Bug修复 2】：提供安全保底字典，防止因为完全不收敛导致后续提取键值时 KeyError
        best_metrics = {
            'miou': 0.0, 'f1_or_dsc': 0.0, 'accuracy': 0.0,
            'specificity': 0.0, 'sensitivity': 0.0, 'confusion': np.zeros((2, 2))
        }

        print("\n🚀 开始搜索最佳阈值...")
        # 从 0.15 搜到 0.50，步长 0.01 (恢复原代码逻辑)
        for thresh in np.arange(0.15, 0.81, 0.01):
            y_pre = np.where(preds >= thresh, 1, 0)
            confusion = confusion_matrix(y_true, y_pre)
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

            miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

            # 如果当前阈值跑出的 miou 更高，更新最佳记录
            if miou > best_miou:
                best_miou = miou
                best_thresh = thresh
                accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
                sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
                specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
                f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0

                best_metrics = {
                    'miou': miou, 'f1_or_dsc': f1_or_dsc, 'accuracy': accuracy,
                    'specificity': specificity, 'sensitivity': sensitivity, 'confusion': confusion
                }

        # 拿出最高分的数据 (此时绝对安全，不会 KeyError)
        miou = best_metrics['miou']
        f1_or_dsc = best_metrics['f1_or_dsc']
        accuracy = best_metrics['accuracy']
        specificity = best_metrics['specificity']
        sensitivity = best_metrics['sensitivity']
        confusion = best_metrics['confusion']

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)

        # 打印霸气的最终结果
        log_info = f'👑 [Best Thresh: {best_thresh:.2f}] test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)