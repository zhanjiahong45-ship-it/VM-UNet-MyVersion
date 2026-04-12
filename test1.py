import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from models.vmunet.vmunetff import VMUNet
from engine import *
import os
import warnings
from utils import *
from configs.config_setting import setting_config

warnings.filterwarnings("ignore")


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


def main(config):
    print('#---------- Preparing Logger ----------#')
    log_dir = os.path.join(config.work_dir, 'log')
    logger = get_logger('test_only', log_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('#---------- Preparing Dataset ----------#')
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False
    )

    criterion = config.criterion
    os.makedirs(os.path.join(config.work_dir, 'outputs'), exist_ok=True)

    # ── 权重路径：优先 mixture.pth，回退 best.pth ─────────────────────────
    ckpt_dir = os.path.dirname(config.best_ckpt_path)
    mixture_path = os.path.join(ckpt_dir, 'mixture.pth')
    best_path    = config.best_ckpt_path

    early_model = None

    if os.path.exists(mixture_path):
        print(f'Loading mixture.pth from {mixture_path}')
        ckpt = torch.load(mixture_path, map_location=device)
        model = build_model(config, device)
        model.load_state_dict(ckpt['model'], strict=False)

        early_model = build_model(config, device)
        early_model.load_state_dict(ckpt['early_model'], strict=False)
        early_model.eval()
        print('early_model loaded from mixture.pth')
    else:
        print(f'mixture.pth not found, loading best.pth from {best_path}')
        ckpt = torch.load(best_path, map_location=device)
        model = build_model(config, device)
        if isinstance(ckpt, dict) and 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    # ─────────────────────────────────────────────────────────────────────

    model.eval()

    test_one_epoch(
        val_loader,
        model,
        criterion,
        logger,
        config,
        test_data_name=config.datasets,
        early_model=early_model,
    )
    print('#---------- 测试完美结束 ----------#')


if __name__ == '__main__':
    config = setting_config
    set_seed(config.seed)
    main(config)