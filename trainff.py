import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
# 更改此处：从新的 vmunetff 导入模型
from models.vmunet.vmunetff import VMUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


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

    print('#----------Set other params----------#')
    # 【核心修改 1】：抛弃 min_loss，改用 max_miou 记录最佳状态
    max_miou = 0.0
    best_loss = 999.0
    best_epoch = 1
    start_epoch = 1

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
        best_epoch = checkpoint.get('best_epoch', 1)
        best_loss = checkpoint.get('best_loss', 999.0)

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, max_miou: {max_miou:.4f}, best_epoch: {best_epoch}, loss: {best_loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer
        )

        # 【核心修改 3】：获取 val_one_epoch 返回的 loss 和 miou
        val_result = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        # 鲁棒性检查：判断 engine.py 是否已经被修改为返回 (loss, miou)
        if isinstance(val_result, tuple):
            loss, current_miou = val_result[0], val_result[1]
        else:
            loss = val_result
            current_miou = 0.0
            print("WARNING: 未检测到 mIoU 返回值，正在使用备用 Loss 判定机制！(请检查 engine.py)")

        # 【核心修改 4】：全新的“唯 mIoU 论”保存逻辑
        is_best = False
        if current_miou > 0:
            if current_miou > max_miou:
                is_best = True
                max_miou = current_miou
                best_loss = loss
                best_epoch = epoch
        else:
            # 兼容性备用方案：如果你还没改 engine.py，它依然能按 loss 跑，不会报错
            if loss < best_loss:
                is_best = True
                best_loss = loss
                best_epoch = epoch

        if is_best:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))

        # 【核心修改 5】：把最佳 mIoU 信息存入 latest.pth 以备断点续训
        torch.save(
            {
                'epoch': epoch,
                'max_miou': max_miou,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            val_loader,
            model,
            criterion,
            logger,
            config,
        )

        # 【核心修改 6】：最终生成的文件名会骄傲地打上 miou 的印记
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-miou{max_miou:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)