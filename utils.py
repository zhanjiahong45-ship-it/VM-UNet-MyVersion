import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt

from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    log_info = f'#----------Config info----------#'
    logger.info(log_info)
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        else:
            log_info = f'{k}: {v},'
            logger.info(log_info)


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                          'SGD'], 'Unsupported optimizer!'

    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(
            model.parameters(),
            lr=config.lr,
            rho=config.rho,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adagrad':
        return torch.optim.Adagrad(
            model.parameters(),
            lr=config.lr,
            lr_decay=config.lr_decay,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            amsgrad=config.amsgrad
        )
    elif config.opt == 'Adamax':
        return torch.optim.Adamax(
            model.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'ASGD':
        return torch.optim.ASGD(
            model.parameters(),
            lr=config.lr,
            lambd=config.lambd,
            alpha=config.alpha,
            t0=config.t0,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'RMSprop':
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            alpha=config.alpha,
            eps=config.eps,
            centered=config.centered,
            weight_decay=config.weight_decay
        )
    elif config.opt == 'Rprop':
        return torch.optim.Rprop(
            model.parameters(),
            lr=config.lr,
            etas=config.etas,
            step_sizes=config.step_sizes,
        )
    elif config.opt == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dampening=config.dampening,
            nesterov=config.nesterov
        )
    else:  # default opt is SGD
        return torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=0.05,
        )


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.milestones,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.gamma,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.mode,
            factor=config.factor,
            patience=config.patience,
            threshold=config.threshold,
            threshold_mode=config.threshold_mode,
            cooldown=config.cooldown,
            min_lr=config.min_lr,
            eps=config.eps
        )
    elif config.sch == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min,
            last_epoch=config.last_epoch
        )
    elif config.sch == 'WP_MultiStepLR':
        lr_func = lambda \
            epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    elif config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
                math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    return scheduler


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    # 处理图像数据
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img

    # 根据数据集处理掩码
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        att_msk = np.squeeze(msk_pred, axis=0)
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    # 设置画布大小
    plt.figure(figsize=(10, 20))

    # 调整子图间距，减少白色边框
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)

    # 原图
    plt.subplot(4, 1, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')

    # Ground Truth 掩码
    plt.subplot(4, 1, 2)
    plt.imshow(msk, cmap='gray')
    plt.axis('off')
    plt.title('Ground Truth Mask')

    # 预测掩码
    plt.subplot(4, 1, 3)
    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')
    plt.title('Predicted Mask')

    # 注意力图谱叠加到原图上
    plt.subplot(4, 1, 4)
    plt.imshow(img)
    plt.imshow(att_msk, cmap='jet', alpha=0.5)  # 热力图叠加到原图上，alpha设置透明度
    plt.axis('off')
    plt.title('Attention Map Overlay')

    # 保存图片
    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class nDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(nDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class CeDiceLoss(nn.Module):
    def __init__(self, num_classes, loss_weight=[0.4, 0.6]):
        super(CeDiceLoss, self).__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.diceloss = nDiceLoss(num_classes)
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_ce = self.celoss(pred, target[:].long())
        loss_dice = self.diceloss(pred, target, softmax=True)
        loss = self.loss_weight[0] * loss_ce + self.loss_weight[1] * loss_dice
        return loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        # True Positives, False Positives, False Negatives
        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        # Tversky 公式：分母中 FN 乘以 alpha，FP 乘以 beta
        Tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        return 1 - Tversky


class BceTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, wb=1, wt=1):
        super(BceTverskyLoss, self).__init__()
        # 复用你 utils.py 中已经写好的 BCELoss
        self.bce = BCELoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.wb = wb
        self.wt = wt

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        tverskyloss = self.tversky(pred, target)

        loss = self.wb * bceloss + self.wt * tverskyloss
        return loss



class GT_BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(GT_BceDiceLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)

    def forward(self, gt_pre, out, target):
        bcediceloss = self.bcedice(out, target)
        gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1 = gt_pre
        gt_loss = self.bcedice(gt_pre5, target) * 0.1 + self.bcedice(gt_pre4, target) * 0.2 + self.bcedice(gt_pre3,
                                                                                                           target) * 0.3 + self.bcedice(
            gt_pre2, target) * 0.4 + self.bcedice(gt_pre1, target) * 0.5
        return bcediceloss + gt_loss


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1), torch.tensor(mask).permute(2, 0, 1)


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        image, mask = data
        return TF.resize(image, [self.size_h, self.size_w]), TF.resize(mask, [self.size_h, self.size_w])


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.hflip(image), TF.hflip(mask)
        else:
            return image, mask


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.vflip(image), TF.vflip(mask)
        else:
            return image, mask


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.angle = random.uniform(degree[0], degree[1])
        self.p = p

    def __call__(self, data):
        image, mask = data
        if random.random() < self.p:
            return TF.rotate(image, self.angle), TF.rotate(mask, self.angle)
        else:
            return image, mask


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            if train:
                self.mean = 157.561
                self.std = 26.706
            else:
                self.mean = 149.034
                self.std = 32.022
        elif data_name == 'isic17':
            if train:
                self.mean = 159.922
                self.std = 28.871
            else:
                self.mean = 148.429
                self.std = 25.748
        elif data_name == 'isic18_82':
            if train:
                self.mean = 156.2899
                self.std = 26.5457
            else:
                self.mean = 149.8485
                self.std = 35.3346

    def __call__(self, data):
        img, msk = data
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - np.min(img_normalized))
                          / (np.max(img_normalized) - np.min(img_normalized))) * 255.
        return img_normalized, msk


from thop import profile  ## 导入thop模块


def cal_params_flops(model, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    print('flops', flops / 1e9)  ## 打印计算量
    print('params', params / 1e6)  ## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))
    logger.info(f'flops: {flops / 1e9}, params: {params / 1e6}, Total params: : {total / 1e6:.4f}')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1, val_or_test=False):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None and val_or_test is True:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        # cv2.imwrite(test_save_path + '/'+case + '.png', prediction*255)
    return metric_list


class FocalLoss(nn.Module):
    """
    针对困难样本（浅色病灶）的 Focal Loss。
    核心逻辑：网络越分不对的像素，惩罚权重呈指数级放大；已经分对的深色病灶，惩罚权重降到极低。
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', smooth=1e-5):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, pred, target):
        size = pred.size(0)
        # 展平以便计算
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        # 【极其关键】：因为 VMUNet 输出已经过了 sigmoid，存在绝对的 0 或 1，
        # 直接求 log 会导致梯度爆炸 (NaN)。必须用 clamp 截断在极小值范围。
        pred_ = torch.clamp(pred_, min=self.smooth, max=1.0 - self.smooth)

        # 计算预测正确的概率 pt
        # 如果 target==1，pt = pred_；如果 target==0，pt = 1 - pred_
        pt = torch.where(target_ == 1, pred_, 1 - pred_)

        # 计算基础的交叉熵 CE = -log(pt)
        bce_loss = -torch.log(pt)

        # 引入类别平衡权重 alpha
        alpha_t = torch.where(target_ == 1, self.alpha, 1 - self.alpha)

        # 核心公式：FL = alpha_t * (1 - pt)^gamma * CE
        # 浅色病灶 (预测概率低，pt小) 的权重 (1-pt)^gamma 会接近 1，Loss 被完整保留
        # 深色病灶 (预测概率高，pt接近1) 的权重 (1-pt)^gamma 会趋近 0，不再产生干扰梯度
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class FocalDiceLoss(nn.Module):
    """
    终极组合损失函数：
    Focal Loss 负责逼迫网络抠出浅色微弱边界；
    Dice Loss 负责保证病灶区域的整体连贯性。
    """

    def __init__(self, alpha=0.25, gamma=2.0, wf=1, wd=1):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        # 复用原有的 DiceLoss
        self.dice = DiceLoss()
        self.wf = wf
        self.wd = wd

    def forward(self, pred, target):
        focalloss = self.focal(pred, target)
        diceloss = self.dice(pred, target)

        # 结合困难样本挖掘与区域连贯性
        loss = self.wf * focalloss + self.wd * diceloss
        return loss
class RobustCompoundLoss(nn.Module):
    """
    Three-component loss designed to simultaneously optimize for:
      1. Overall segmentation quality (FocalDice — your existing loss)
      2. Difficult sample emphasis (OHEM — online hard example mining)
      3. Light-lesion boundary recall (PixelConfidencePenalty)

    Usage:
        criterion = RobustCompoundLoss(alpha=0.75, gamma=2.0)
    """
    def __init__(self, alpha=0.75, gamma=2.0, ohem_ratio=0.7, confidence_band=(0.3, 0.7)):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ohem_ratio = ohem_ratio
        self.low = confidence_band[0]
        self.high = confidence_band[1]

    def focal_loss_per_pixel(self, pred, target):
        """Returns per-pixel focal loss (no reduction)"""
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal = alpha_t * ((1 - pt) ** self.gamma) * bce
        return focal

    def dice_loss(self, pred, target):
        """Standard batch-level dice loss"""
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def forward(self, pred, target):
        # ── Component 1: Focal loss with OHEM ──
        focal_per_pixel = self.focal_loss_per_pixel(pred, target)  # (B, 1, H, W)

        # OHEM: keep only the hardest ohem_ratio fraction of pixels
        focal_flat = focal_per_pixel.view(focal_per_pixel.shape[0], -1)  # (B, H*W)
        num_pixels = focal_flat.shape[1]
        k = max(int(self.ohem_ratio * num_pixels), 1)

        # Per-sample topk, then average
        topk_vals, _ = torch.topk(focal_flat, k, dim=1)
        focal_ohem = topk_vals.mean()

        # ── Component 2: Dice loss (unchanged, for global overlap) ──
        dice = self.dice_loss(pred, target)

        # ── Component 3: Confidence-band penalty ──
        # Penalize pixels where prediction is uncertain (0.3–0.7)
        # These are precisely the light-colored lesion boundaries the model is unsure about
        uncertain_mask = ((pred > self.low) & (pred < self.high)).float()
        # How many uncertain pixels fall on true positives? (= missed light lesions)
        uncertain_on_positive = uncertain_mask * target
        # Penalty: push these uncertain-positive pixels toward 1.0
        if uncertain_on_positive.sum() > 0:
            confidence_penalty = F.binary_cross_entropy(
                pred * uncertain_on_positive,
                target * uncertain_on_positive,
                reduction='sum'
            ) / (uncertain_on_positive.sum() + 1e-6)
        else:
            confidence_penalty = torch.tensor(0.0, device=pred.device)

        # ── Final weighted combination ──
        # Weights: Focal-OHEM dominates early, confidence penalty grows with training
        total = 1.0 * focal_ohem + 1.0 * dice + 0.5 * confidence_penalty

        return total


class GentleCompoundLoss(nn.Module):
    """验证过的 FocalDiceLoss + 轻微的不确定性惩罚"""

    def __init__(self, alpha=0.75, gamma=2.0, confidence_weight=0.1):
        super().__init__()
        # 基础损失函数
        self.base_loss = FocalDiceLoss(alpha=alpha, gamma=gamma, wf=1, wd=1)
        self.confidence_weight = confidence_weight

    def forward(self, pred, target):
        base = self.base_loss(pred, target)

        # 对称置信度惩罚：惩罚所有不确定的像素（预测值在 0.3 到 0.7 之间）
        uncertain_mask = ((pred > 0.3) & (pred < 0.7)).float()

        if uncertain_mask.sum() > 0:
            penalty = F.binary_cross_entropy(
                pred, target, reduction='none'
            )
            penalty = (penalty * uncertain_mask).sum() / (uncertain_mask.sum() + 1e-6)
        else:
            penalty = 0.0

        return base + self.confidence_weight * penalty