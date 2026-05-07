from torchvision import transforms
from utils import *
from datetime import datetime

class setting_config_finetune:
    """LoRA Decoder 微调配置 —— 基于 Epoch 8 保护浅色病灶，过滤毛发"""

    network = 'vmunet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 2, 2, 1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': None,
    }

    datasets = 'isic18'
    train_path = '/root/root/VM-UNet/data/isic18/train'
    val_path = '/root/root/VM-UNet/data/val'
    data_path = './data/isic18/'

    criterion = BceTverskyLoss(wb=1, wt=1, alpha=0.3, beta=0.7)

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 16
    seed = 42
    world_size = None
    rank = None
    amp = True
    gpu_id = '0'
    batch_size = 32

    # ============== 修改点 1：从 Epoch 8 出发 ==============
    # 请将这里替换为你实际的 epoch 8 权重路径
    best_ckpt_path = '/root/root/VM-UNet/results/vmunet_isic18_Sunday_03_May_2026_20h_44m_16s/checkpoints/early_epochs/epoch_008_miou0.7661.pth'

    # ============== 修改点 2：少量 Epoch ==============
    epochs = 60

    # 工作目录命名调整
    work_dir = ('results/' + network + '_' + datasets + '_LoRA_Decoder_'
                + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/')

    print_interval = 20
    val_interval = 1
    save_interval = 100
    threshold = 0.50
    only_test_and_save_figs = False
    img_save_path = ''

    preprocess = CLAHEGammaPreprocess(gamma=1.5, clip_limit=5.0)

    train_transformer = transforms.Compose([
        preprocess,  # <--- 第一步先做 CLAHE + Gamma
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])

    test_transformer = transforms.Compose([
        preprocess,  # <--- 推理/验证也必须做
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'AdamW'
    if opt == 'AdamW':
        # ============== 修改点 3：极小学习率 ==============
        lr = 1e-4
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-4 # 加一点正则化防止 LoRA 过拟合
        amsgrad = False

    sch = 'WP_CosineLR'
    if sch == 'WP_CosineLR':
        warm_up_epochs = 1
        T_max = epochs
        eta_min = 1e-7