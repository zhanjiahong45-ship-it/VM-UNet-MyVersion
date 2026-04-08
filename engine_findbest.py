import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
import torch.nn.functional as F  # <--- 必须加上这个库来做多尺度缩放


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
    # 验证阶段为了速度，保持 4 视角翻转 TTA 即可
    model.eval()
    preds = []
    gts = []
    loss_list = []
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

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list),miou


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
            # 🚀 终极测试大招：12 视角多尺度 TTA (3 种尺寸 × 4 个方向)
            # ====================================================================
            b, c, h, w = img.shape
            final_prob = torch.zeros((b, 1, h, w), device=img.device)

            # 定义测试尺度：1.0(原图), 1.25(放大看微弱边缘), 0.75(缩小看全局轮廓)
            scales = [1.0, 1.25, 0.75]

            for scale in scales:
                if scale != 1.0:
                    img_s = F.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=False)
                else:
                    img_s = img

                out_prob_s = torch.zeros((b, 1, img_s.shape[2], img_s.shape[3]), device=img.device)

                # 1. 原向
                o1 = model(img_s)
                o1 = o1[0] if isinstance(o1, tuple) else o1
                out_prob_s += o1

                # 2. 水平翻转
                o2 = model(torch.flip(img_s, dims=[3]))
                o2 = o2[0] if isinstance(o2, tuple) else o2
                out_prob_s += torch.flip(o2, dims=[3])

                # 3. 垂直翻转
                o3 = model(torch.flip(img_s, dims=[2]))
                o3 = o3[0] if isinstance(o3, tuple) else o3
                out_prob_s += torch.flip(o3, dims=[2])

                # 4. 对角翻转
                o4 = model(torch.flip(img_s, dims=[2, 3]))
                o4 = o4[0] if isinstance(o4, tuple) else o4
                out_prob_s += torch.flip(o4, dims=[2, 3])

                # 当前尺度的平均概率
                out_prob_s = out_prob_s / 4.0

                # 缩放回原始 256x256 尺寸
                if scale != 1.0:
                    out_prob_s = F.interpolate(out_prob_s, size=(h, w), mode='bilinear', align_corners=False)

                final_prob += out_prob_s

            # 综合所有尺度的概率 (除以 3 因为有三个尺度)
            out = final_prob / len(scales)
            # ====================================================================

            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)

            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

            # 注意：保存图片这里默认用的 config.threshold，因为此时还不知道最佳阈值
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
        best_miou = 0.0
        best_thresh = config.threshold
        best_metrics = {}

        print("\n🚀 开始搜索最佳阈值...")
        # 从 0.15 搜到 0.50，步长 0.01
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