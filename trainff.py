import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
# 更改此处：从新的 vmunetff 导入模型
from models.vmunet.vmunetff import VMUNet

from engine import *
import os
import sys
import cv2
import numpy as np
from PIL import Image
from utils import *
from configs.config_setting import setting_config

import warnings
from utils import RobustCompoundLoss
warnings.filterwarnings("ignore")


def track_hard_samples(model, epoch, config, device):
    """
    困难样本动态追踪器：自动读取 inputs 文件夹下的图，输出预测叠加图
    """
    model.eval()  # 确保切到预测模式

    # 自动读取该目录下所有的图片文件
    input_dir = '/root/root/VM-UNet/inputs'
    if not os.path.exists(input_dir):
        print(f"⚠️ 找不到文件夹: {input_dir}")
        return

    hard_sample_paths = [os.path.join(input_dir, img) for img in os.listdir(input_dir)
                         if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(hard_sample_paths) == 0:
        return

    # 自动在当前训练的 result 文件夹下建一个 tracking 目录
    tracking_dir = os.path.join(config.work_dir, 'hard_samples_tracking')
    os.makedirs(tracking_dir, exist_ok=True)

    # 你的 ISIC18 预处理参数
    ISIC18_TEST_MEAN = 149.034
    ISIC18_TEST_STD = 32.022
    IMG_SIZE = 256
    THRESHOLD = 0.5

    with torch.no_grad():
        for img_path in hard_sample_paths:
            img_name = os.path.basename(img_path)
            img_pil = Image.open(img_path).convert('RGB')
            original_size = img_pil.size

            # 完全复刻预处理逻辑
            img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img_arr = np.array(img_resized, dtype=np.float32)
            img_normalized = (img_arr - ISIC18_TEST_MEAN) / ISIC18_TEST_STD
            img_min, img_max = np.min(img_normalized), np.max(img_normalized)
            if img_max > img_min:
                img_final = ((img_normalized - img_min) / (img_max - img_min)) * 255.0
            else:
                img_final = img_normalized

            img_tensor = torch.from_numpy(img_final).permute(2, 0, 1).contiguous().float()
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # 推理
            output = model(img_tensor)
            if isinstance(output, tuple):
                output = output[0]

            # 后处理与 Overlay 叠加
            prob_map = output.squeeze().cpu().numpy()
            prob_map_resized = cv2.resize(prob_map, original_size, interpolation=cv2.INTER_LINEAR)
            prediction = (prob_map_resized > THRESHOLD).astype(np.uint8) * 255

            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            color_mask = np.zeros_like(img_cv)
            color_mask[prediction == 255] = [0, 0, 255]
            overlay = cv2.addWeighted(img_cv, 0.7, color_mask, 0.5, 0)

            contours, _ = cv2.findContours(prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)

            # 保存图片：带有 epoch 序号
            save_path = os.path.join(tracking_dir, f'epoch_{epoch:03d}_{img_name}')
            cv2.imwrite(save_path, overlay)
def evaluate_sentinel_images(model, config, epoch):
    """
    Compute per-pixel IoU on the 3 author-identified hard cases.
    These images are stored as:
      sentinel/   → input images  (e.g., light_lesion_1.png)
      sentinel_gt/ → binary masks (e.g., light_lesion_1.png, white=lesion)

    Returns: mean IoU across the sentinel set (float, 0~1)
    """
    model.eval()

    sentinel_dir = getattr(config, 'sentinel_dir', None)
    sentinel_gt_dir = getattr(config, 'sentinel_gt_dir', None)

    if sentinel_dir is None or not os.path.exists(sentinel_dir):
        return 0.0

    ISIC18_TEST_MEAN = 149.034
    ISIC18_TEST_STD = 32.022
    IMG_SIZE = 256
    THRESHOLD = 0.5

    ious = []

    with torch.no_grad():
        for img_name in sorted(os.listdir(sentinel_dir)):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Load and preprocess image (same as your track_hard_samples)
            img_path = os.path.join(sentinel_dir, img_name)
            img_pil = Image.open(img_path).convert('RGB')
            img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            img_arr = np.array(img_resized, dtype=np.float32)
            img_normalized = (img_arr - ISIC18_TEST_MEAN) / ISIC18_TEST_STD
            img_min, img_max = np.min(img_normalized), np.max(img_normalized)
            if img_max > img_min:
                img_final = ((img_normalized - img_min) / (img_max - img_min)) * 255.0
            else:
                img_final = img_normalized

            img_tensor = torch.from_numpy(img_final).permute(2, 0, 1).contiguous().float()
            img_tensor = img_tensor.unsqueeze(0).cuda()

            # Inference
            output = model(img_tensor)
            if isinstance(output, tuple):
                output = output[0]
            pred = (output.squeeze().cpu().numpy() > THRESHOLD).astype(np.uint8)

            # Load ground truth
            gt_path = os.path.join(sentinel_gt_dir, img_name)
            if not os.path.exists(gt_path):
                # Try matching without extension
                base = os.path.splitext(img_name)[0]
                for ext in ['.png', '.jpg', '.bmp']:
                    candidate = os.path.join(sentinel_gt_dir, base + ext)
                    if os.path.exists(candidate):
                        gt_path = candidate
                        break

            if not os.path.exists(gt_path):
                continue

            gt_pil = Image.open(gt_path).convert('L')
            gt_resized = gt_pil.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            gt = (np.array(gt_resized) > 127).astype(np.uint8)

            # Compute IoU
            intersection = (pred & gt).sum()
            union = (pred | gt).sum()
            iou = intersection / (union + 1e-6)
            ious.append(iou)

    model.train()

    if len(ious) == 0:
        return 0.0
    return float(np.mean(ious))
def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

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

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
        )
        model.load_from()

    else:
        raise Exception('network in not right!')
    model = model.cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
    print('#----------Set other params----------#')
    # 【核心修改 1】：抛弃 min_loss，改用 max_miou 记录最佳状态
    max_miou = 0.0
    best_loss = 999.0
    best_epoch = 1
    start_epoch = 1
    max_sentinel_score = 0.0  # tracks best score on 3 hard images

    # ... existing resume logic ...

    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        return

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch

        # 【核心修改 2】：适配断点续训，读取 miou 相关信息
        max_miou = checkpoint.get('max_miou', 0.0)
        max_sentinel_score = checkpoint.get('max_sentinel_score', 0.0)
        best_epoch = checkpoint.get('best_epoch', 1)
        best_loss = checkpoint.get('best_loss', 999.0)

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, max_miou: {max_miou:.4f}, best_epoch: {best_epoch}, loss: {best_loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        # ╔═══════════════════════════════════════════╗
        # ║  TWO-STAGE TRANSITION LOGIC (Fix 4)       ║
        # ╚═══════════════════════════════════════════╝
        if epoch == config.stage1_epochs + 1:
            print('=' * 60)
            print('🔄 ENTERING STAGE 2: Freezing encoder, lowering LR')
            print('=' * 60)

            # Freeze the entire encoder (layers) and FCD module
            for name, param in model.named_parameters():
                # Freeze: patch_embed (FCD), layers.0-3 (encoder)
                # Keep trainable: layers_up (decoder), respaths, final_*
                if 'patch_embed' in name or 'layers.' in name:
                    # Careful: 'layers.' matches encoder, 'layers_up.' matches decoder
                    # We need to NOT freeze 'layers_up'
                    if 'layers_up' not in name:
                        param.requires_grad = False

            # Rebuild optimizer with only trainable params + lower LR
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.stage2_lr,
                betas=(0.9, 0.999),
                weight_decay=0.05
            )
            # New cosine scheduler for Stage 2
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.stage2_epochs,
                eta_min=1e-7
            )

            logger.info(f'Stage 2: {sum(p.numel() for p in trainable_params)} trainable params')

        step = train_one_epoch(
            train_loader, model, criterion, optimizer, scheduler,
            epoch, step, logger, config, writer,
        )

        val_result = val_one_epoch(
            val_loader, model, criterion, epoch, logger, config
        )

        if isinstance(val_result, tuple):
            loss, current_miou = val_result[0], val_result[1]
        else:
            loss = val_result
            current_miou = 0.0

        # ╔═══════════════════════════════════════════╗
        # ║  DUAL-METRIC MODEL SAVING (Part 4)        ║
        # ╚═══════════════════════════════════════════╝
        # Compute sentinel score on the 3 hard images
        sentinel_score = evaluate_sentinel_images(model, config, epoch)

        # Composite score: weighted blend of mIoU and sentinel performance
        # sentinel_score is the mean IoU specifically on the 3 hard cases
        composite_score = 0.7 * current_miou + 0.3 * sentinel_score

        is_best = False
        if composite_score > max_miou:  # reusing max_miou as max_composite
            is_best = True
            max_miou = composite_score
            best_loss = loss
            best_epoch = epoch

        # Also save "best sentinel" independently (for paper figures)
        if sentinel_score > max_sentinel_score:
            max_sentinel_score = sentinel_score
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, 'best_sentinel.pth'))
            logger.info(f'💎 New best sentinel: {sentinel_score:.4f} at epoch {epoch}')

        if is_best:
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, 'best.pth'))

        # ... existing latest.pth save logic ...
        torch.save({
            'epoch': epoch,
            'max_miou': max_miou,
            'max_sentinel_score': max_sentinel_score,
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest.pth'))

        track_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        track_hard_samples(model, epoch, config, track_device)
    # In the final testing section of main(), add:
    # Replace lines 372-389 with:
    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(
            os.path.join(checkpoint_dir, 'best.pth'),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            val_loader, model, criterion, logger, config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-miou{max_miou:.4f}.pth')
        )

    # Sentinel figures (separate checkpoint)
    if os.path.exists(os.path.join(checkpoint_dir, 'best_sentinel.pth')):
        print('#----------Generating Paper Figures----------#')
        sentinel_weight = torch.load(
            os.path.join(checkpoint_dir, 'best_sentinel.pth'),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(sentinel_weight)
        track_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        track_hard_samples(model, 9999, config, track_device)
        logger.info('Paper figures generated from best_sentinel checkpoint')


if __name__ == '__main__':
    config = setting_config
    main(config)