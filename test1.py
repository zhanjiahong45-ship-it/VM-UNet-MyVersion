import torch
from torch.utils.data import DataLoader
from datasets.dataset import NPY_datasets
from models.vmunet.vmunet1 import VMUNet1, FocalDiceLoss
from engine import *
import os
import warnings
from utils import *
from configs.config_setting import setting_config

warnings.filterwarnings("ignore")


def main(config):
    print('#---------- Preparing Logger ----------#')
    log_dir = os.path.join(config.work_dir, 'log')
    logger = get_logger('test_only', log_dir)
    log_config_info(config, logger)

    print('#---------- Preparing Model ----------#')
    # 初始化模型结构
    model = VMUNet1(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        depths=config.model_config['depths'],
        depths_decoder=config.model_config['depths_decoder'],
        drop_path_rate=config.model_config['drop_path_rate'],
        load_ckpt_path=config.model_config['load_ckpt_path'],
    )
    model = model.cuda()

    print('#---------- Preparing Dataset ----------#')
    # 加载验证集/测试集数据
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 测试时 batch_size 设为 1 最稳妥
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False
    )

    # 损失函数（仅测试时作参考用，不影响指标计算）
    criterion = FocalDiceLoss()

    # ========================================================
    # 核心：智能查找最优权重路径（彻底解决路径不一致找不到文件的问题）
    # =======================================================


    best_ckpt_path = '/root/root/VM-UNet/results/vmunet_isic18_25_February_23h_48m_57s/checkpoints/best.pth'


    # 加载权重
    best_weight = torch.load(best_ckpt_path, map_location=torch.device('cpu'))

    # 兼容各种不同版本的权重字典保存格式
    # 兼容各种不同版本的权重字典保存格式
    if 'model_state_dict' in best_weight:
        model.load_state_dict(best_weight['model_state_dict'], strict=False)
    elif 'model' in best_weight:
        model.load_state_dict(best_weight['model'], strict=False)
    else:
        model.load_state_dict(best_weight, strict=False)

    print("✅ 成功加载 300 轮训练中的最优权重！开始计算大满贯指标...")
    os.makedirs(os.path.join(config.work_dir, 'outputs'), exist_ok=True)

    # ========================================================
    # 核心黑科技：模型包装器（拦截并拆包元组输出，完美骗过 engine.py）
    # ========================================================
    class ModelWrapper(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

        def forward(self, x):
            out = self.base_model(x)
            # 如果输出是元组 (final_out, fcd_out)，只返回最终的分割结果
            if isinstance(out, tuple):
                return out[0]
            return out

    # 把我们带壳的模型实例化
    test_model = ModelWrapper(model)
    test_model.eval()

    # 直接调用 engine.py 里的测试函数，传入包装好的 test_model！
    test_one_epoch(
        val_loader,
        test_model,  # <--- 注意这里传的是带壳的模型
        criterion,
        logger,
        config,
        test_data_name=config.datasets
    )
    print('#---------- 测试完美结束 ----------#')


if __name__ == '__main__':
    config = setting_config
    set_seed(config.seed)
    main(config)