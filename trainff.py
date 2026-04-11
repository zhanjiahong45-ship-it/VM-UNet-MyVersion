import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import os
import sys
import numpy as np
import warnings
from PIL import Image
from models.vmunet.vmunetff import VMUNet
from datasets.dataset import NPY_datasets
from engine import train_one_epoch, val_one_epoch
from utils import *
from configs.config_setting import setting_config

warnings.filterwarnings("ignore")


def inpaint_with_skin(img_tensor, prob, threshold=0.4):
    anchor = (prob > threshold).float()
    non_anchor = 1.0 - anchor
    skin_color = (img_tensor * non_anchor).sum(dim=[2, 3], keepdim=True) / \
                 (non_anchor.sum(dim=[2, 3], keepdim=True) + 1e-8)
    return img_tensor * (1 - anchor) + skin_color.expand_as(img_tensor) * anchor


def build_model(config, device):
    m = VMUNet(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        depths=config.model_config['depths'],
        depths_decoder=config.model_config['depths_decoder'],
        drop_path_rate=config.model_config['drop_path_rate'],
        load_ckpt_path=None,
    ).to(device)
    return m


def select_best_early_model(ckpt_paths, config, device):
    """
    从 epoch2/3/4 权重中选出对浅色病灶最慷慨的模型。
    评估方式：对训练集前20张图做 ghost 增强，比较各模型在 GT 正样本区域的平均概率。
    """
    from utils import myFaintLesionAugmentor
    augmentor = myFaintLesionAugmentor()

    # 取训练集前20张图
    train_images_dir = os.path.join(config.data_path, 'train/images/')
    train_masks_dir = os.path.join(config.data_path, 'train/masks/')
    img_names = sorted(os.listdir(train_images_dir))[:20]

    best_score = -1.0
    best_path = ckpt_paths[0]

    for ckpt_path in ckpt_paths:
        if not os.path.exists(ckpt_path):
            continue
        model = build_model(config, device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        scores = []
        with torch.no_grad():
            for name in img_names:
                img = np.array(Image.open(os.path.join(train_images_dir, name)).convert('RGB'))
                msk = np.array(Image.open(os.path.join(train_masks_dir, name)).convert('L'))
                msk = np.expand_dims(msk, -1)
                msk_bin = (msk > 127).astype(np.float32)

                # ghost 增强
                img_faint, _ = augmentor.apply_specific_mode((img, msk_bin), mode='ghost')

                dummy_mask = np.zeros((config.input_size_h, config.input_size_w, 1), dtype=np.float32)
                img_t, _ = config.test_transformer((img_faint, dummy_mask))
                img_t = img_t.unsqueeze(0).float().to(device)

                out = model(img_t)
                out = out[0] if isinstance(out, tuple) else out

                # GT 正样本区域平均概率
                msk_t, _ = config.test_transformer((img, msk_bin))
                gt = (msk_t > 0.5).float().to(device)
                if gt.sum() > 0:
                    scores.append((out * gt).sum().item() / gt.sum().item())

        score = np.mean(scores) if scores else 0.0
        print(f"  Early model {os.path.basename(ckpt_path)}: faint_score={score:.4f}")
        if score > best_score:
            best_score = score
            best_path = ckpt_path

    print(f"  Selected: {os.path.basename(best_path)} (score={best_score:.4f})")
    return best_path


def track_hard_samples(model, epoch, config, device, early_model=None):
    model.eval()

    input_dir = getattr(config, 'hard_samples_input_dir', './inputs/')
    if not os.path.exists(input_dir):
        return

    hard_sample_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not hard_sample_paths:
        return

    tracking_dir = os.path.join(config.work_dir, 'hard_samples_tracking')
    os.makedirs(tracking_dir, exist_ok=True)

    with torch.no_grad():
        for img_path in hard_sample_paths:
            img_name = os.path.basename(img_path)
            original_img = Image.open(img_path).convert('RGB')
            original_img = original_img.resize((config.input_size_w, config.input_size_h))
            img_np = np.array(original_img)
            dummy_mask = np.zeros((config.input_size_h, config.input_size_w, 1), dtype=np.float32)
            img_tensor, _ = config.test_transformer((img_np, dummy_mask))
            img_tensor = img_tensor.unsqueeze(0).float().to(device)

            # 最终模型第一遍
            out_final = model(img_tensor)
            out_final = out_final[0] if isinstance(out_final, tuple) else out_final
            prob_max = out_final.max().item()
            prob_mean = out_final.mean().item()

            is_faint = prob_max > 0.5 and prob_mean < 0.05
            print(f"  [{img_name}] max={prob_max:.3f} mean={prob_mean:.4f} faint={is_faint}")

            # 面板数量取决于是否有早期模型且触发浅色病灶
            if early_model is not None and is_faint:
                fig, axes = plt.subplots(1, 7, figsize=(35, 5))

                def to_np_img(t):
                    arr = t.squeeze().permute(1, 2, 0).cpu().numpy()
                    return np.clip(arr, 0, 1)

                # 1. 原图
                axes[0].imshow(img_np); axes[0].set_title("1.Original"); axes[0].axis('off')

                # 2. best模型切原图 → Mask-m
                mask_m = out_final
                axes[1].imshow(mask_m.squeeze().cpu().numpy(), cmap='jet')
                axes[1].set_title(f"2.Mask-m(best)\nmax={mask_m.max().item():.2f}"); axes[1].axis('off')

                # 3. 早期模型切原图 → Mask-e
                early_model.eval()
                out_e = early_model(img_tensor)
                out_e = out_e[0] if isinstance(out_e, tuple) else out_e
                mask_e = out_e
                axes[2].imshow(mask_e.squeeze().cpu().numpy(), cmap='jet')
                axes[2].set_title(f"3.Mask-e(early)\nmax={mask_e.max().item():.2f}"); axes[2].axis('off')

                # 4. 用Mask-m补色 → m1
                img_m1 = inpaint_with_skin(img_tensor, mask_m, threshold=0.4)
                axes[3].imshow(to_np_img(img_m1)); axes[3].set_title("4.m1(inpaint by m)"); axes[3].axis('off')

                # 5. 用Mask-e补色 → me
                img_me = inpaint_with_skin(img_tensor, mask_e, threshold=0.4)
                axes[4].imshow(to_np_img(img_me)); axes[4].set_title("5.me(inpaint by e)"); axes[4].axis('off')

                # 6. m1跑早期模型
                out_m1 = early_model(img_m1)
                out_m1 = out_m1[0] if isinstance(out_m1, tuple) else out_m1
                axes[5].imshow(out_m1.squeeze().cpu().numpy(), cmap='jet')
                axes[5].set_title(f"6.early(m1)\nmax={out_m1.max().item():.2f}"); axes[5].axis('off')

                # 7. me跑早期模型
                out_me = early_model(img_me)
                out_me = out_me[0] if isinstance(out_me, tuple) else out_me
                axes[6].imshow(out_me.squeeze().cpu().numpy(), cmap='jet')
                axes[6].set_title(f"7.early(me)\nmax={out_me.max().item():.2f}"); axes[6].axis('off')

            else:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis('off')
                prob_f = out_final.squeeze().cpu().numpy()
                axes[1].imshow(prob_f, cmap='jet')
                axes[1].set_title(f"Ep{epoch} max={prob_max:.2f} mean={prob_mean:.3f}")
                axes[1].axis('off')
                axes[2].imshow(img_np); axes[2].imshow(prob_f, cmap='jet', alpha=0.5)
                axes[2].set_title("Overlay"); axes[2].axis('off')

            save_path = os.path.join(tracking_dir, f"epoch_{epoch:03d}_{img_name}")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    outputs = os.path.join(config.work_dir, 'outputs')
    for d in [checkpoint_dir, outputs]:
        os.makedirs(d, exist_ok=True)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, pin_memory=True, num_workers=config.num_workers)

    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=config.num_workers, drop_last=True)

    model = VMUNet(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        depths=config.model_config['depths'],
        depths_decoder=config.model_config['depths_decoder'],
        drop_path_rate=config.model_config['drop_path_rate'],
        load_ckpt_path=config.model_config['load_ckpt_path'],
    )
    model.load_from()
    model = model.to(device)

    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = torch.cuda.amp.GradScaler() if config.amp else None

    max_miou = 0.0
    best_epoch = 0
    step = 0
    early_model = None  # 选出的早期模型，epoch5后赋值

    for epoch in range(1, config.epochs + 1):
        torch.cuda.empty_cache()

        # epoch5：切换到 1N 数据集，loss 保持 BceDiceLoss 不变
        if epoch == 5:
            train_dataset.set_phase(2)
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                      shuffle=True, pin_memory=True, num_workers=config.num_workers)
            logger.info("Phase 2: 切换到 1N 数据集")

            # 选出最佳早期模型
            early_ckpts = [os.path.join(checkpoint_dir, f'epoch_{i:03d}.pth') for i in [2, 3, 4]]
            best_early_path = select_best_early_model(early_ckpts, config, device)
            early_model = build_model(config, device)
            early_model.load_state_dict(torch.load(best_early_path, map_location=device))
            early_model.eval()
            logger.info(f"早期模型已选定: {best_early_path}")

        train_loss = train_one_epoch(
            train_loader=train_loader, model=model, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, epoch=epoch,
            step=step, logger=logger, config=config, writer=writer, scaler=scaler
        )

        val_loss, current_miou = val_one_epoch(
            val_loader=val_loader, model=model, criterion=criterion,
            epoch=epoch, logger=logger, config=config, writer=writer
        )

        if current_miou > max_miou:
            max_miou = current_miou
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            logger.info(f'New best mIoU: {max_miou:.4f} at epoch {epoch}')

        # 保存 epoch 2/3/4 权重供早期模型选择
        if 2 <= epoch <= 4:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'epoch_{epoch:03d}.pth'))
            logger.info(f'Saved early checkpoint: epoch_{epoch:03d}.pth')

        # epoch5 之后才追踪困难样本
        if epoch >= 5:
            try:
                track_hard_samples(model, epoch, config, device, early_model=early_model)
            except Exception as e:
                logger.warning(f"困难样本追踪失败: {e}")

    logger.info(f'Training Complete! Best mIoU: {max_miou:.4f} at epoch {best_epoch}')


if __name__ == '__main__':
    config = setting_config
    main(config)