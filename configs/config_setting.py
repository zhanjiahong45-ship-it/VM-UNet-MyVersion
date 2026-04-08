from torchvision import transforms
from utils import *
from datetime import datetime


class setting_config:
    """
    the config of training setting.
    """

    network = 'vmunet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        'depths': [2, 2, 9, 2],
        'depths_decoder': [2, 2, 2, 1],
        # 👑 回调 1：恢复 0.2！释放模型被压抑的拟合能力，让它去学精细边缘
        'drop_path_rate': 0.2,
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
    }

    datasets = 'isic18'
    if datasets == 'isic18':
        data_path = './data/isic18/'
    elif datasets == 'isic17':
        data_path = './data/isic17/'
    else:
        raise Exception('datasets in not right!')

    criterion = GentleCompoundLoss(alpha=0.75, gamma=2.0, confidence_weight=0.1)

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 4
    seed = 42
    world_size = None
    rank = None
    amp = True
    gpu_id = '0'
    batch_size = 32

    # 👑 回调 2：设定为 120 轮。前 60 轮冲刺，后 60 轮在低学习率下精细沉降
    stage1_epochs = 80  # Stage 1: full model training
    stage2_epochs = 40  # Stage 2: decoder-focused refinement
    epochs = 120  # total (must equal stage1 + stage2)

    # Stage 2 LR (10x lower than stage 1 peak)
    stage2_lr = 2e-5

    # ── Sentinel Metric Config (Part 4) ──
    # Paths to the 3 author-identified hard cases (place in /root/root/VM-UNet/sentinel/)
    sentinel_dir = '/root/root/VM-UNet/sentinel'
    # These are the ground truth masks for the 3 sentinel images
    sentinel_gt_dir = '/root/root/VM-UNet/sentinel_gt'

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 1
    save_interval = 100
    threshold = 0.50
    only_test_and_save_figs = False

    best_ckpt_path = '/root/root/VM-UNet/results/9_8288/checkpoints/best-epoch76-miou0.8288.pth'
    img_save_path = ''

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])

    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                   'SGD'], 'Unsupported optimizer!'

    if opt == 'AdamW':
        # 👑 回调 3：恢复 2e-4，给足模型冲刺 83 分的动力
        lr = 2e-4
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.05
        amsgrad = False

    sch = 'WP_CosineLR'

    if sch == 'WP_CosineLR':
        warm_up_epochs = 10
        # 👑 核心绝杀：T_max 对齐 120。
        # 这样在第 60 轮（之前容易过拟合的拐点），学习率刚好衰减到一半，强行按住模型不让它乱飞。
        T_max = 80
        eta_min = 1e-6