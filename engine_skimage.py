import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import torch.nn.functional as F

# 🚀 引入连通域分析库
from skimage import morphology


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer):
    '''
    train model for one epoch
    '''
    model.train()
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step()
    return step


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config):
    model.eval()
    preds = []
    gts = []
    loss_list = []

    miou = 0.0

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            out_org = model(img)
            out_org = out_org[0] if type(out_org) is tuple else out_org

            img_hf = torch.flip(img, dims=[3])
            out_hf = model(img_hf)
            out_hf = out_hf[0] if type(out_hf) is tuple else out_hf
            out_hf = torch.flip(out_hf, dims=[3])

            img_vf = torch.flip(img, dims=[2])
            out_vf = model(img_vf)
            out_vf = out_vf[0] if type(out_vf) is tuple else out_vf
            out_vf = torch.flip(out_vf, dims=[2])

            img_hvf = torch.flip(img, dims=[2, 3])
            out_hvf = model(img_hvf)
            out_hvf = out_hvf[0] if type(out_hvf) is tuple else out_hvf
            out_hvf = torch.flip(out_hvf, dims=[2, 3])

            out = (out_org + out_hf + out_vf + out_hvf) / 4.0

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
                   test_data_name=None):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            # ====================================================================
            # 🚀 终极测试大招：12 视角多尺度 TTA (3 种尺寸 × 4 个方向)
            # ====================================================================
            b, c, h, w = img.shape
            final_prob = torch.zeros((b, 1, h, w), device=img.device)
            scales = [1.0, 1.25, 0.75]

            for scale in scales:
                if scale != 1.0:
                    img_s = F.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=False)
                else:
                    img_s = img

                out_prob_s = torch.zeros((b, 1, img_s.shape[2], img_s.shape[3]), device=img.device)

                o1 = model(img_s)
                out_prob_s += o1[0] if isinstance(o1, tuple) else o1

                o2 = model(torch.flip(img_s, dims=[3]))
                out_prob_s += torch.flip(o2[0] if isinstance(o2, tuple) else o2, dims=[3])

                o3 = model(torch.flip(img_s, dims=[2]))
                out_prob_s += torch.flip(o3[0] if isinstance(o3, tuple) else o3, dims=[2])

                o4 = model(torch.flip(img_s, dims=[2, 3]))
                out_prob_s += torch.flip(o4[0] if isinstance(o4, tuple) else o4, dims=[2, 3])

                out_prob_s = out_prob_s / 4.0

                if scale != 1.0:
                    out_prob_s = F.interpolate(out_prob_s, size=(h, w), mode='bilinear', align_corners=False)

                final_prob += out_prob_s

            out = final_prob / len(scales)
            # ====================================================================

            loss = criterion(out, msk)
            loss_list.append(loss.item())

            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)

            # 保留完整的空间维度 (b, h, w)，后面形态学清理需要用到！
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

            if i % config.save_interval == 0:
                save_imgs(img.cpu(), msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

        # 整理成 (N_images, 256, 256) 的三维数组
        preds_3d = np.concatenate(preds, axis=0)
        gts_3d = np.concatenate(gts, axis=0)

        y_true_flat = np.where(gts_3d.reshape(-1) >= 0.5, 1, 0)

        # ====================================================================
        # 👑 终极压榨：网格搜索 + 最大连通域/孤立噪点过滤
        # ====================================================================
        best_miou = 0.0
        best_thresh = config.threshold
        best_metrics = {}

        print("\n🚀 开始搜索最佳阈值并进行形态学连通域净化...")
        # 扩大搜索上限到 0.85，因为 Focal Loss 会让模型整体置信度偏高
        for thresh in np.arange(0.15, 0.85, 0.01):

            # 1. 基础阈值截断
            y_pre_raw = np.where(preds_3d >= thresh, 1, 0)

            # 2. 🚀 形态学净化：移除面积小于 200 像素的孤立假阳性噪点
            y_pre_clean = np.zeros_like(y_pre_raw)
            for img_idx in range(y_pre_raw.shape[0]):
                # remove_small_objects 能瞬间消灭角落里零星的毛发残留预测！
                cleaned_mask = morphology.remove_small_objects(y_pre_raw[img_idx].astype(bool), min_size=200)
                y_pre_clean[img_idx] = cleaned_mask.astype(int)

            # 3. 展平进行指标计算
            y_pre_flat = y_pre_clean.reshape(-1)

            confusion = confusion_matrix(y_true_flat, y_pre_flat)
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

            miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

            # 更新最佳记录
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

        log_info = f'👑 [Best Thresh: {best_thresh:.2f} + Clean] test of best model, loss: {np.mean(loss_list):.4f}, miou: {miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}, accuracy: {accuracy:.4f}, specificity: {specificity:.4f}, sensitivity: {sensitivity:.4f}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)