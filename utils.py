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
import cv2
from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric
import copy
import numpy as np
import cv2
import random
import math


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



class myAdvancedSkinCutout:
    """
    Advanced cutout augmentation for skin-lesion segmentation.

    Features
    --------
    - Random shape selection: axis-aligned rectangles, ellipses, or
      convex polygons (3-6 vertices).
    - Stochastic "donut" (hole-in-hole) nesting that restores original
      pixels inside an outer fill region, creating island / ring patterns.
    - Size clamping via *max_size_ratio* to prevent total lesion occlusion.
    - Only the **image** is modified; the ground-truth mask is returned
      unchanged.

    Parameters
    ----------
    p : float
        Probability of applying the augmentation at all.
    n_holes : tuple[int, int]
        (min, max) number of primary holes per call.
    max_size_ratio : float
        Maximum hole dimension as a fraction of the corresponding image
        dimension.  Keeps holes from covering the entire lesion.
    fill_value : float
        Constant pixel value used to fill primary (outer) holes.
    donut_prob : float
        Per-layer probability of spawning a nested inner hole that
        restores original pixels (or alternates fill / restore).
    max_nested : int
        Maximum nesting depth.  Each successive layer alternates between
        *restoring* original pixels and *filling* with ``fill_value``,
        producing multi-ring donut patterns when > 1.
    min_hole_px : int
        Minimum hole side length in pixels.  Prevents degenerate shapes
        when *max_size_ratio* × image size is very small.
    """

    def __init__(
            self,
            p: float = 0.5,
            n_holes: tuple = (1, 3),
            max_size_ratio: float = 0.2,
            fill_value: float = 157.56,
            donut_prob: float = 0.5,
            max_nested: int = 1,
            min_hole_px: int = 10,
    ):
        self.p = p
        self.n_holes = n_holes
        self.max_size_ratio = max_size_ratio
        self.fill_value = fill_value
        self.donut_prob = donut_prob
        self.max_nested = max_nested
        self.min_hole_px = min_hole_px

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _random_shape_mask(
            self, h: int, w: int, hs: int, ws: int, y1: int, x1: int
    ) -> np.ndarray:
        """Return a binary uint8 mask (same size as the image) with a single
        randomly chosen filled shape inside the bounding box
        ``(x1, y1, x1+ws, y1+hs)``.

        Polygon vertices are convex-hull ordered so ``cv2.fillPoly`` always
        produces a valid, non-self-intersecting polygon.
        """
        mask = np.zeros((h, w), dtype=np.uint8)
        shape = random.choice(("rect", "ellipse", "poly"))

        if shape == "rect":
            cv2.rectangle(mask, (x1, y1), (x1 + ws, y1 + hs), 1, -1)

        elif shape == "ellipse":
            cx, cy = x1 + ws // 2, y1 + hs // 2
            # Random rotation for extra variety
            angle = random.uniform(0, 360)
            cv2.ellipse(mask, (cx, cy), (ws // 2, hs // 2), angle, 0, 360, 1, -1)

        else:  # polygon
            n_verts = random.randint(3, 6)
            pts = np.array(
                [
                    [random.randint(x1, x1 + ws), random.randint(y1, y1 + hs)]
                    for _ in range(n_verts)
                ],
                dtype=np.int32,
            )
            # Convex-hull ordering avoids self-intersecting polygons
            hull = cv2.convexHull(pts)
            cv2.fillPoly(mask, [hull], 1)

        return mask

    def _safe_range(self, dim: int) -> tuple:
        """Return (lo, hi) for ``random.randint`` clamped to valid bounds."""
        lo = self.min_hole_px
        hi = max(lo, int(dim * self.max_size_ratio))
        return lo, hi

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------
    def __call__(self, data: tuple) -> tuple:
        """
        Parameters
        ----------
        data : (np.ndarray, np.ndarray)
            ``(image, mask)`` pair where *image* has shape ``(H, W, C)``
            and *mask* is the ground-truth segmentation label.

        Returns
        -------
        (np.ndarray, np.ndarray)
            Augmented image and the **unmodified** mask.
        """
        image, mask_label = data

        if random.random() > self.p:
            return image, mask_label

        h, w = image.shape[:2]
        h_lo, h_hi = self._safe_range(h)
        w_lo, w_hi = self._safe_range(w)

        # Nothing useful to draw if limits are degenerate
        if h_hi < h_lo or w_hi < w_lo:
            return image, mask_label

        orig_image = image.copy()
        num_holes = random.randint(*self.n_holes)

        for _ in range(num_holes):
            # --- primary (outer) hole ---
            hs = random.randint(h_lo, h_hi)
            ws = random.randint(w_lo, w_hi)
            y1 = random.randint(0, h - hs)
            x1 = random.randint(0, w - ws)

            outer = self._random_shape_mask(h, w, hs, ws, y1, x1)
            image[outer == 1] = self.fill_value

            # --- nested donut layers ---
            cur_hs, cur_ws = hs, ws
            cur_y1, cur_x1 = y1, x1
            restore = True  # first inner layer restores original pixels

            for depth in range(self.max_nested):
                if random.random() >= self.donut_prob:
                    break

                # Shrink to 30-60 % of the parent (randomised)
                shrink = random.uniform(0.3, 0.6)
                cur_hs = int(cur_hs * shrink)
                cur_ws = int(cur_ws * shrink)
                if cur_hs < self.min_hole_px or cur_ws < self.min_hole_px:
                    break

                # Centre inside the parent bounding box
                cur_y1 = y1 + (hs - cur_hs) // 2
                cur_x1 = x1 + (ws - cur_ws) // 2

                inner = self._random_shape_mask(h, w, cur_hs, cur_ws, cur_y1, cur_x1)

                if restore:
                    image[inner == 1] = orig_image[inner == 1]
                else:
                    image[inner == 1] = self.fill_value

                restore = not restore  # alternate each layer

        return image, mask_label


class AdaptiveFocalDiceLoss(nn.Module):
    """
    随 Epoch 自适应增加权重的 Focal Dice Loss。
    - 早期 (低 Gamma)：平等学习，建立全局认知。
    - 后期 (高 Gamma)：疯狂惩罚被遗忘的困难/浅色样本。
    """

    def __init__(self, gamma_start=0.5, gamma_end=3.0, total_epochs=120, wd=1.0):
        super().__init__()
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.total_epochs = total_epochs
        self.wd = wd
        self.dice = DiceLoss()
        self.current_gamma = gamma_start  # 初始状态

    def update_epoch(self, epoch):
        # 供外部在每个 epoch 开始前调用，动态推高 gamma
        self.current_gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * (epoch / self.total_epochs)

    def forward(self, pred, target):
        # 防 NaN 截断
        pred_ = torch.clamp(pred, min=1e-5, max=1.0 - 1e-5)

        # 极其优雅的 pt 计算：exp(-BCE) 刚好等于正确类别的预测概率 pt
        bce = F.binary_cross_entropy(pred_, target, reduction='none')
        pt = torch.exp(-bce)

        # 动态 Focal 计算
        focal_loss = ((1 - pt) ** self.current_gamma * bce).mean()
        dice_loss = self.dice(pred, target)

        return focal_loss + self.wd * dice_loss


class ModelEMA:
    """
    模型权重的指数移动平均 (EMA) 封装器。
    用于保留前几轮对浅色病灶的敏锐嗅觉。
    """

    def __init__(self, model, decay=0.998):
        self.decay = decay
        # 深拷贝一个不参与梯度计算的模型
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # 在每个 Step 结束后调用，缓慢融合新权重
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


class FaintLesionAwareLoss(nn.Module):
    """
    Combined loss that maintains sensitivity to faint / small / donut lesions
    throughout training.

    Components
    ----------
    1. **Logit-space Focal BCE** — recovers gradient magnitude that sigmoid
       compression destroys for confident predictions.
    2. **Area-reweighted Dice** — each sample's Dice contribution is scaled
       by ``1 / sqrt(foreground_area)``, so a lesion covering 3% of pixels
       gets ~3x the gradient of one covering 30%.
    3. **Sample-level hard mining** — the per-sample total loss is sorted;
       only the hardest ``hard_ratio`` fraction contributes to the backward
       pass.  This ensures faint-lesion images (which have higher loss) are
       never washed out by the easy majority.
    4. **Boundary loss** — a Sobel-based edge extractor from the GT mask
       computes a boundary-only BCE, giving the optimizer an explicit signal
       on faint edges.

    Parameters
    ----------
    gamma_start, gamma_end, total_epochs : float
        Adaptive focal exponent schedule (same interface as your existing
        ``AdaptiveFocalDiceLoss``).
    hard_ratio : float
        Fraction of samples (per batch) to keep after hard mining.
        0.7 means the easiest 30% of samples are dropped each step.
    boundary_weight : float
        Coefficient on the boundary loss term.
    dice_weight : float
        Coefficient on the area-reweighted Dice term.
    """

    def __init__(
            self,
            gamma_start: float = 0.5,
            gamma_end: float = 3.0,
            total_epochs: int = 120,
            hard_ratio: float = 0.7,
            boundary_weight: float = 0.5,
            dice_weight: float = 1.0,
    ):
        super().__init__()
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.total_epochs = total_epochs
        self.hard_ratio = hard_ratio
        self.boundary_weight = boundary_weight
        self.dice_weight = dice_weight
        self.current_gamma = gamma_start

        # Sobel kernels for boundary extraction (registered as buffers so
        # they move to GPU automatically with .cuda())
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    # ------------------------------------------------------------------
    # Public interface (matches AdaptiveFocalDiceLoss)
    # ------------------------------------------------------------------
    def update_epoch(self, epoch: int):
        progress = min(epoch / self.total_epochs, 1.0)
        self.current_gamma = (
                self.gamma_start + (self.gamma_end - self.gamma_start) * progress
        )

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------
    def _prob_to_logit(self, p: torch.Tensor) -> torch.Tensor:
        """Invert sigmoid so we can use logit-space BCE."""
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p / (1 - p))

    def _focal_bce_per_sample(
            self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Focal BCE computed in logit space, reduced to per-sample scalar.

        Returns shape ``(B,)``.
        """
        logits = self._prob_to_logit(pred)
        # Numerically stable BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.current_gamma * bce
        # Reduce spatial dims → per-sample
        return focal.view(focal.size(0), -1).mean(dim=1)

    def _area_reweighted_dice_per_sample(
            self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Dice loss where each sample is weighted by 1/sqrt(foreground_area).

        Returns shape ``(B,)``.
        """
        smooth = 1.0
        B = pred.size(0)
        p = pred.view(B, -1)
        t = target.view(B, -1)
        intersection = (p * t).sum(dim=1)
        dice = (2 * intersection + smooth) / (p.sum(1) + t.sum(1) + smooth)
        dice_loss = 1 - dice  # (B,)

        # Area weight: small lesions → big weight
        fg_area = t.sum(dim=1).clamp(min=1.0)
        total_pixels = t.size(1)
        # Inverse sqrt so a 3%-area lesion gets ~3x weight vs 30%-area
        weight = torch.sqrt(total_pixels / fg_area)
        # Normalize weights so they average to 1 (no scale change)
        weight = weight / (weight.mean() + 1e-8)

        return dice_loss * weight

    def _boundary_loss(
            self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """BCE computed only on GT boundary pixels (Sobel edge detection).

        Falls back to zero if no boundary pixels exist.
        """
        # Extract boundary from GT mask
        # target shape: (B, 1, H, W) — Sobel needs this
        if target.dim() == 3:
            target_4d = target.unsqueeze(1)
        else:
            target_4d = target

        gx = F.conv2d(target_4d, self.sobel_x.to(target_4d.device), padding=1)
        gy = F.conv2d(target_4d, self.sobel_y.to(target_4d.device), padding=1)
        edge = (gx.abs() + gy.abs()).squeeze(1)  # (B, H, W)
        boundary_mask = (edge > 0.1).float()

        n_boundary = boundary_mask.sum()
        if n_boundary < 1:
            return torch.tensor(0.0, device=pred.device)

        # Squeeze pred to match boundary_mask shape
        p = pred.squeeze(1) if pred.dim() == 4 else pred
        t = target.squeeze(1) if target.dim() == 4 else target

        logits = self._prob_to_logit(p)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        return (bce * boundary_mask).sum() / n_boundary

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # --- Per-sample losses ---
        focal_per_sample = self._focal_bce_per_sample(pred, target)
        dice_per_sample = self._area_reweighted_dice_per_sample(pred, target)

        combined_per_sample = focal_per_sample + self.dice_weight * dice_per_sample

        # --- Sample-level hard mining ---
        B = combined_per_sample.size(0)
        k = max(int(self.hard_ratio * B), 1)
        topk_vals, _ = torch.topk(combined_per_sample, k)
        main_loss = topk_vals.mean()

        # --- Boundary loss ---
        b_loss = self._boundary_loss(pred, target)

        return main_loss + self.boundary_weight * b_loss



class myFaintLesionAugmentor:
    """
    myFaintLesionAugmentor — production version.

    Six pattern modes:
    diffuse    — whole lesion fades toward skin, patchy/uneven
    donut      — faded ring + preserved random center
    nested     — recursive donut-in-donut (2-3 layers)
    multi      — 2-4 disconnected faded blobs
    shattered  — 5-15 overlapping holes fragmenting the lesion
    ghost      — COMPLETE fadeout, lesion becomes nearly invisible
    original   — return the original image without any fading
    """

    def __init__(
            self,
            p: float = 0.75,
            fade_range: tuple = (0.3, 0.95),
            mode_probs: dict = None,
    ):
        self.p = p
        self.fade_range = fade_range
        self.mode_probs = mode_probs or {
            'diffuse': 0.15,
            'donut': 0.15,
            'nested': 0.15,
            'multi': 0.15,
            'shattered': 0.25,
            'ghost': 0.15,
        }

    # ==================================================================
    # 👑 新增：直接指定模式应用 (跳过概率投骰子，供 Dataset 底层调用)
    # ==================================================================
    def apply_specific_mode(self, data, mode):
        image, mask = data

        # 1. 原图模式直接返回
        if mode == 'original':
            return image, mask

        # 2. 提取 mask 和边界
        m = self._get_mask_bin(mask)
        ys, xs = np.where(m > 0.5)
        if len(ys) < 100:
            return image, mask
        bb = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))

        h, w = image.shape[:2]
        was_uint8 = image.dtype == np.uint8
        img = image.astype(np.float32)

        # 3. 获取皮肤底色 (Skin color from non-lesion pixels)
        skin_mask = 1.0 - m
        skin_color = np.zeros(3, dtype=np.float32)
        if skin_mask.sum() > 50:
            for c in range(3):
                skin_color[c] = (img[:, :, c] * skin_mask).sum() / (skin_mask.sum() + 1e-8)
        else:
            skin_color[:] = img.mean(axis=(0, 1))

        # 4. 匹配并生成对应模式的 fade_mask
        self._ghost_override = False
        generators = {
            'diffuse': self._gen_diffuse,
            'donut': self._gen_donut,
            'nested': self._gen_nested,
            'multi': self._gen_multi,
            'shattered': self._gen_shattered,
            'ghost': self._gen_ghost,
        }

        # 如果传入了不存在的模式，默认使用 diffuse
        generator_func = generators.get(mode, self._gen_diffuse)
        fade_mask = generator_func(h, w, m, bb)

        # 5. 纹理与强度计算 (Patchy texture & Fade strength)
        if random.random() < 0.7:
            fade_mask = self._patchy(fade_mask, h, w)

        if mode == 'ghost' or self._ghost_override:
            strength = random.uniform(0.85, 0.98)
        else:
            strength = random.uniform(*self.fade_range)

        # 6. 图像混合应用 (Blend lesion toward skin color)
        fade_3d = (fade_mask * strength)[:, :, np.newaxis]
        skin_img = np.ones_like(img) * skin_color[np.newaxis, np.newaxis, :]
        result = img * (1 - fade_3d) + skin_img * fade_3d

        if was_uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)

        # mask 保持不变，强制模型学习隐秘特征
        return result, mask

    # ==================================================================
    # Shape primitives — fully randomized
    # ==================================================================
    def _rand_shape(self, h, w, cy, cx, ry, rx):
        """One random filled shape. Type, vertex count, rotation all random."""
        mask = np.zeros((h, w), dtype=np.float32)
        kind = random.choice(['ellipse', 'poly', 'blob'])

        # Clamp center and radii to valid ranges
        ry, rx = max(3, int(ry)), max(3, int(rx))
        cy = int(np.clip(cy, ry, h - ry - 1))
        cx = int(np.clip(cx, rx, w - rx - 1))

        if kind == 'ellipse':
            angle = random.uniform(0, 360)
            cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)

        elif kind == 'poly':
            n = random.randint(3, 8)
            pts = []
            for _ in range(n):
                px = random.randint(max(0, cx - rx), min(w - 1, cx + rx))
                py = random.randint(max(0, cy - ry), min(h - 1, cy + ry))
                pts.append([px, py])
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            cv2.fillPoly(mask, [hull], 1.0)

        else:  # blob — deformed ellipse with jagged boundary
            n = random.randint(10, 20)
            roughness = random.uniform(0.2, 0.6)
            pts = []
            for i in range(n):
                a = 2 * math.pi * i / n
                rf = max(0.3, 1.0 + random.gauss(0, roughness))
                px = int(np.clip(cx + rx * rf * math.cos(a), 0, w - 1))
                py = int(np.clip(cy + ry * rf * math.sin(a), 0, h - 1))
                pts.append([px, py])
            hull = cv2.convexHull(np.array(pts, dtype=np.int32))
            cv2.fillPoly(mask, [hull], 1.0)

        # Random edge softness
        blur = max(3, random.choice([3, 5, 7, 9, 11, 15])) | 1
        return cv2.GaussianBlur(mask, (blur, blur), 0)

    def _rand_pos_in_bbox(self, bb, margin=0.3):
        """Random center within or near the lesion bbox."""
        y1, x1, y2, x2 = bb
        ry, rx = (y2 - y1) / 2, (x2 - x1) / 2
        cy = random.randint(int(y1 - ry * margin), int(y2 + ry * margin))
        cx = random.randint(int(x1 - rx * margin), int(x2 + rx * margin))
        return cy, cx

    def _rand_size(self, ry, rx, lo=0.1, hi=0.6):
        """Random size as fraction of lesion dimensions."""
        s = random.uniform(lo, hi)
        return max(4, int(ry * s)), max(4, int(rx * s))

    # ==================================================================
    # Texture: patchy non-uniform fade
    # ==================================================================
    def _patchy(self, mask, h, w):
        """Modulate fade mask with low-frequency noise for uneven fading."""
        scale = random.choice([8, 12, 16, 24])
        noise = np.random.rand(h // scale + 1, w // scale + 1).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        k = random.choice([15, 21, 31]) | 1
        noise = cv2.GaussianBlur(noise, (k, k), 0)
        lo = random.uniform(0.3, 0.6)
        noise = lo + (1 - lo) * (noise - noise.min()) / (noise.ptp() + 1e-8)
        return mask * noise

    # ==================================================================
    # Lesion constraint: keep fade within/near the lesion
    # ==================================================================
    def _constrain(self, fade, mask_bin, bleed=0.1):
        """Limit fade region to lesion area with optional slight bleed."""
        dilate_k = random.randint(1, 4)
        dilated = cv2.dilate(mask_bin, np.ones((5, 5)), iterations=dilate_k)
        dilated = cv2.GaussianBlur(dilated, (11, 11), 0)
        boundary = mask_bin * (1 - bleed) + dilated * bleed
        return fade * np.clip(boundary + 0.05, 0, 1)

    # ==================================================================
    # Six pattern generators
    # ==================================================================
    def _gen_diffuse(self, h, w, m, bb):
        """Whole lesion fades, patchy and uneven."""
        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
        expand = random.uniform(0.9, 1.4)
        blob = self._rand_shape(h, w, cy, cx,
                                int(ry * expand), int(rx * expand))
        return self._constrain(blob, m, bleed=random.uniform(0, 0.2))

    def _gen_donut(self, h, w, m, bb):
        """Faded ring, random-shaped hole in the center."""
        outer = self._gen_diffuse(h, w, m, bb)

        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        iy, ix = self._rand_size(ry, rx, 0.15, 0.5)
        oy = random.randint(-max(1, ry // 3), max(1, ry // 3))
        ox = random.randint(-max(1, rx // 3), max(1, rx // 3))
        inner = self._rand_shape(h, w, cy + oy, cx + ox, iy, ix)

        return np.clip(outer - inner * random.uniform(0.6, 1.0), 0, 1)

    def _gen_nested(self, h, w, m, bb):
        """Recursive donut-in-donut: 2-3 concentric layers."""
        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        n_layers = random.randint(2, 3)
        result = np.zeros((h, w), dtype=np.float32)
        cur_ry, cur_rx = ry, rx
        fade_on = True  # alternates: fade, restore, fade, restore...

        for layer in range(n_layers):
            scale = random.uniform(0.5, 0.8) if layer > 0 else random.uniform(0.9, 1.3)
            cur_ry = max(5, int(cur_ry * scale))
            cur_rx = max(5, int(cur_rx * scale))

            if cur_ry < 5 or cur_rx < 5:
                break

            oy = random.randint(-max(1, cur_ry // 4), max(1, cur_ry // 4))
            ox = random.randint(-max(1, cur_rx // 4), max(1, cur_rx // 4))
            ring = self._rand_shape(h, w, cy + oy, cx + ox, cur_ry, cur_rx)

            if fade_on:
                result = np.maximum(result, ring)
            else:
                result = np.clip(result - ring * random.uniform(0.5, 1.0), 0, 1)

            fade_on = not fade_on

        return self._constrain(result, m, bleed=0.1)

    def _gen_multi(self, h, w, m, bb):
        """2-4 disconnected faded blobs."""
        y1, x1, y2, x2 = bb
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
        combined = np.zeros((h, w), dtype=np.float32)

        for _ in range(random.randint(2, 4)):
            cy, cx = self._rand_pos_in_bbox(bb, margin=0.2)
            sy, sx = self._rand_size(ry, rx, 0.15, 0.5)
            blob = self._rand_shape(h, w, cy, cx, sy, sx)
            combined = np.maximum(combined, blob * random.uniform(0.5, 1.0))

        return self._constrain(combined, m, bleed=0.15)

    def _gen_shattered(self, h, w, m, bb):
        """5-15 dense overlapping holes fragmenting the lesion."""
        y1, x1, y2, x2 = bb
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
        combined = np.zeros((h, w), dtype=np.float32)

        n_holes = random.randint(5, 15)
        for _ in range(n_holes):
            hy = random.randint(y1, y2)
            hx = random.randint(x1, x2)
            sy, sx = self._rand_size(ry, rx, 0.04, 0.25)
            hole = self._rand_shape(h, w, hy, hx, sy, sx)
            combined = np.maximum(combined, hole * random.uniform(0.4, 1.0))

        # Strictly within lesion
        blurred = cv2.GaussianBlur(m, (7, 7), 0)
        return combined * blurred

    def _gen_ghost(self, h, w, m, bb):
        """
        Complete fadeout — entire lesion becomes nearly invisible.
        """
        y1, x1, y2, x2 = bb
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
        ry, rx = (y2 - y1) // 2, (x2 - x1) // 2

        expand = random.uniform(1.0, 1.3)
        blob = self._rand_shape(h, w, cy, cx,
                                int(ry * expand), int(rx * expand))
        fade = self._constrain(blob, m, bleed=0.05)

        # Override to extreme fade
        self._ghost_override = True
        return fade

    # ==================================================================
    # Legacy __call__ (保留以兼容旧的普通 transform 模式)
    # ==================================================================
    def _get_mask_bin(self, mask):
        m = mask[:, :, 0] if mask.ndim == 3 else mask
        return (m > (127 if m.max() > 1 else 0.5)).astype(np.float32)

    def __call__(self, data):
        image, mask = data

        if random.random() > self.p:
            return image, mask

        # Pick random mode based on prob dictionary
        r = random.random()
        cum = 0
        mode = 'diffuse'
        for name, prob in self.mode_probs.items():
            cum += prob
            if r < cum:
                mode = name
                break

        # 委托给新的 apply_specific_mode 处理
        return self.apply_specific_mode(data, mode)
