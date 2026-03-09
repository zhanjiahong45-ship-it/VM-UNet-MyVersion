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
        'drop_path_rate': 0.2,  # [保留原样]
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
    }

    datasets = 'isic18'
    if datasets == 'isic18':
        data_path = './data/isic18/'
    elif datasets == 'isic17':
        data_path = './data/isic17/'
    else:
        raise Exception('datasets in not right!')

    # ✅ 必须修改 1：1:1 平权，解决敏感度低下的偏科问题
    criterion = BceDiceLoss(wb=1, wd=1)

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
    epochs = 300

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    # ✅ 必须修改 2：每一轮都评估，坚决不漏掉真正的最高 mIoU
    val_interval = 1
    save_interval = 100
    threshold = 0.50
    only_test_and_save_figs = True
    best_ckpt_path = '/root/root/VM-UNet/results/vmunet_isic18_Sunday_08_March_2026_18h_16m_37s/checkpoints/best.pth'
    img_save_path = ''

    # [保留原样]：不引入任何新的复杂数据增强
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
        # ✅ 必须修改 3：降速防震荡，适配 BS=32
        lr = 2e-4
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.05
        amsgrad = False

    sch = 'WP_CosineLR' # [保留原样]

    if sch == 'WP_CosineLR':
        # ✅ 配合 lr 的降低，将 warmup 缩短到 10 轮
        warm_up_epochs = 10
        T_max = 300
        eta_min = 1e-5