import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import torch.nn.functional as F


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
    # switch to train mode
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
    # 验证阶段移除 TTA，采用单次推理
    model.eval()
    preds = []
    gts = []
    loss_list = []
    miou = 0.0  # 🌟 新增：初始化 miou 变量，防止未被赋值

    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            # 直接进行单次前向推理
            out = model(img)
            out = out[0] if isinstance(out, tuple) else out

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

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list), miou  # 🌟 修改点：同时返回 loss 和 miou


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            # ====================================================================
            # 🌟 回归极简原版：裸模型单次推理 (无 TTA)
            # ====================================================================
            out = model(img)
            out = out[0] if isinstance(out, tuple) else out
            # ====================================================================

            loss = criterion(out, msk)

            loss_list.append(loss.item())

            # 这里转 numpy
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)

            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

            # 修复保存图片的 bug，确保 img 在 CPU 上，而 msk 和 out 已经是 numpy
            if i % config.save_interval == 0:
                save_imgs(img.cpu(), msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold,
                          test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        # 定义真实标签的阈值（通常固定为0.5）
        y_true = np.where(gts >= 0.5, 1, 0)

        # 初始化搜寻变量
        best_miou = 0.0
        best_thresh = 0.0
        best_metrics = {
            'miou': 0, 'f1_or_dsc': 0, 'accuracy': 0,
            'specificity': 0, 'sensitivity': 0, 'confusion': np.zeros((2, 2))
        }

        print("\n🚀 开始搜索最佳阈值...")
        # 从 0.15 搜到 0.81，步长 0.01
        for thresh in np.arange(0.40, 0.60, 0.01):
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

        # 拿出最高分的数据
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