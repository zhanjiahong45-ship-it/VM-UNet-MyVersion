import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter

# 导入 V3 模型
from models.vmunet.vmunet_skip3 import VMUNet

from engine import *
import os
import sys
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils import *
# 导入配置
from configs.config_setting3 import setting_config

import warnings

warnings.filterwarnings("ignore")


# 边缘生成函数 (Sobel)
def generate_edge_tensor(label):
    # label: (B, 1, H, W)
    device = label.device
    # Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)

    edge_x = F.conv2d(label, sobel_x, padding=1)
    edge_y = F.conv2d(label, sobel_y, padding=1)

    edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
    # 二值化
    edge = (edge > 0.1).float()
    return edge


def main(config):
    # 更新目录名
    config.work_dir = config.work_dir[:-1] + "_skip3_forced/"

    # 双重保险：强制覆盖 T_max
    config.T_max = config.epochs

    print('#----------Creating logger----------#')
    if not os.path.exists(config.work_dir): os.makedirs(config.work_dir)
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs): os.makedirs(outputs)

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
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Preparing Model (V3)----------#')
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    model.load_from()
    model = model.cuda()

    cal_params_flops(model, 256, logger)

    print('#----------Strategy: Freezing Backbone----------#')
    # [策略3] 冻结主干
    for name, param in model.named_parameters():
        if "skip_gdms" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.eta_min)

    criterion = config.criterion
    criterion_edge = torch.nn.BCELoss()

    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    print('#----------Training Start----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        # [策略3] 解冻 (Epoch 51)
        if epoch == 51:
            print(f"[{epoch}] >>> Unfreezing Backbone! Full Fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True

            # 使用较小 LR 进行全局微调
            optimizer = optim.AdamW(model.parameters(), lr=config.lr * 0.5, weight_decay=0.02)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs - 50,
                                                             eta_min=config.eta_min)

        torch.cuda.empty_cache()

        # --- Train Loop ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch}")

        # [关键修正] 解包元组
        for i, (images, labels) in enumerate(pbar):
            # [核心修复] 强制转换为 float (float32)
            images = images.cuda().float()
            labels = labels.cuda().float()

            # 维度检查 (B, H, W) -> (B, 1, H, W)
            if labels.ndim == 3:
                labels = labels.unsqueeze(1)

            optimizer.zero_grad()

            outputs, edge_preds = model(images)  # Unpack tuple

            loss_seg = criterion(outputs, labels)

            # Edge Loss
            gt_edge = generate_edge_tensor(labels)
            loss_edge = 0
            for pred in edge_preds:
                pred_up = F.interpolate(pred, size=gt_edge.shape[2:], mode='bilinear', align_corners=False)
                loss_edge += criterion_edge(pred_up, gt_edge)
            if len(edge_preds) > 0: loss_edge /= len(edge_preds)

            # Total Loss (权重 0.1)
            loss = loss_seg + 0.1 * loss_edge

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(
                {'loss': loss.item(), 'edge': loss_edge.item() if isinstance(loss_edge, torch.Tensor) else 0})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # --- Val Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                # [核心修复] 验证集也要转 float
                images = images.cuda().float()
                labels = labels.cuda().float()

                if labels.ndim == 3:
                    labels = labels.unsqueeze(1)

                outputs, _ = model(images)  # Ignore edges during val
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = avg_val_loss
            min_epoch = epoch
            print(f">>> Best Model Saved: {min_loss:.4f}")

        if epoch % config.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'latest.pth'))


if __name__ == '__main__':
    config = setting_config
    main(config)