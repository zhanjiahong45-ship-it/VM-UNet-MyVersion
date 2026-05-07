import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from datasets.dataset_cr import NPY_datasets_CR              # ← 新增
from tensorboardX import SummaryWriter
from models.vmunet.vmunetff import VMUNet

from engine import *
from engine_cr import train_one_epoch_cr                      # ← 新增
import os
import sys

from utils import *
from consistency_loss import ConsistencyLoss                  # ← 新增
from configs.config_setting import setting_config

import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# def track_hard_samples(model, epoch, config, device):
#     """
#     极简版：单次推理，生成 3 面板对比图 (原图, 概率图, 叠加图)
#     """
#     model.eval()
#
#     input_dir = getattr(config, 'hard_samples_input_dir', './inputs/')
#     if not os.path.exists(input_dir):
#         return
#
#     hard_sample_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
#                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     if not hard_sample_paths:
#         return
#
#     tracking_dir = os.path.join(config.work_dir, 'hard_samples_tracking')
#     os.makedirs(tracking_dir, exist_ok=True)
#
#     with torch.no_grad():
#         for img_path in hard_sample_paths:
#             img_name = os.path.basename(img_path)
#             original_img = Image.open(img_path).convert('RGB')
#             original_img = original_img.resize((config.input_size_w, config.input_size_h))
#             img_np = np.array(original_img)
#             dummy_mask = np.zeros((config.input_size_h, config.input_size_w, 1), dtype=np.float32)
#             img_tensor, _ = config.test_transformer((img_np, dummy_mask))
#             img_tensor = img_tensor.unsqueeze(0).float().to(device)
#
#             # 仅仅跑一遍模型
#             out = model(img_tensor)
#             out = out[0] if isinstance(out, tuple) else out
#             prob_max = out.max().item()
#             prob_mean = out.mean().item()
#
#             # 绘制 3 面板图
#             fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#             axes[0].imshow(img_np)
#             axes[0].set_title("Original")
#             axes[0].axis('off')
#
#             prob_f = out.squeeze().cpu().numpy()
#             axes[1].imshow(prob_f, cmap='jet')
#             axes[1].set_title(f"Ep{epoch} max={prob_max:.2f} mean={prob_mean:.3f}")
#             axes[1].axis('off')
#
#             axes[2].imshow(img_np)
#             axes[2].imshow(prob_f, cmap='jet', alpha=0.5)
#             axes[2].set_title("Overlay")
#             axes[2].axis('off')
#
#             save_path = os.path.join(tracking_dir, f"epoch_{epoch:03d}_{img_name}")
#             plt.savefig(save_path, bbox_inches='tight')
#             plt.close()


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('#----------Preparing dataset----------#')
    # 🌟 训练集换成 CR 版本（每个样本返回 (img_normal, img_aug, msk) 三元组）
    train_dataset = NPY_datasets_CR(
        config.data_path, config, train=True,
        fade_p=0.75,                  # 75% 概率触发病灶褪色
        fade_range=(0.3, 0.95),       # 褪色强度范围
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    # 🌟 val 仍然用原 NPY_datasets（CR 只在 train 时用）
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
    # 🌟 新增：一致性损失（ConDSeg 风格，threshold 与 config.threshold 对齐）
    consistency_loss_fn = ConsistencyLoss(threshold=config.threshold)
    # 🌟 一致性损失权重，可调，0.5 是推荐起始值
    lambda_cons = getattr(config, 'lambda_cons', 0.5)
    logger.info(f'[CR] consistency_loss_fn={consistency_loss_fn}, lambda_cons={lambda_cons}')

    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    max_miou = 0.0  # 🌟 追踪 mIoU
    start_epoch = 1
    best_epoch = 1

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
        max_miou, best_epoch = checkpoint.get('max_miou', 0.0), checkpoint.get('best_epoch', 1)

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, max_miou: {max_miou:.4f}, best_epoch: {best_epoch}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        # 🌟 用 CR 版的 train_one_epoch
        step = train_one_epoch_cr(
            train_loader,
            model,
            criterion,
            consistency_loss_fn,        # ← 新增的一致性损失
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            lambda_cons=lambda_cons,    # ← 新增超参
        )

        # val 不变（仍是单路 forward）
        val_loss, current_miou = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        # 🌟 基于 mIoU 保存最佳模型
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

        # 🌟 新增：前 20 epoch 每个独立保存（用于轨迹分析）
        if epoch <= 20:
            early_dir = os.path.join(checkpoint_dir, 'early_epochs')
            os.makedirs(early_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(early_dir, f'epoch_{epoch:03d}_miou{current_miou:.4f}.pth')
            )
            logger.info(f'[Early] Saved epoch_{epoch:03d}_miou{current_miou:.4f}.pth')

        # 🌟 极简版困难样本追踪
        # try:
        #     track_hard_samples(model, epoch, config, device)
        # except Exception as e:
        #     logger.warning(f"困难样本追踪失败: {e}")

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-miou{max_miou:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)