import torch
from models.vmunet.vmunetff import VMUNet

model = VMUNet(num_classes=1, input_channels=3,
               depths=[2, 2, 9, 2], depths_decoder=[2, 2, 2, 1])

ckpt_path = '/root/root/VM-UNet/results/vmunet_isic18_Monday_04_May_2026_11h_44m_39s/checkpoints/best.pth'  # 改成你的实际路径
ckpt = torch.load(ckpt_path, map_location='cpu')

# 兼容不同保存格式
if isinstance(ckpt, dict) and 'model' in ckpt:
    state = ckpt['model']
elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
    state = ckpt['state_dict']
else:
    state = ckpt

# 过滤掉 thop/fvcore 留下的统计 buffer
state = {k: v for k, v in state.items()
         if 'total_ops' not in k and 'total_params' not in k}

missing, unexpected = model.load_state_dict(state, strict=False)
print(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
if unexpected:
    print("unexpected:", unexpected[:5], "..." if len(unexpected) > 5 else "")

# 现在能 print gate 值了
print()
print("=== Local gate values ===")
model.vmunet.report_local_gates()