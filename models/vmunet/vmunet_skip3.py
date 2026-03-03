from .vmamba_skip3 import VSSM
import torch
from torch import nn


class VMUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes

        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate,
                           )

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # [修改] 返回元组
        logits, edge_preds = self.vmunet(x)

        if self.num_classes == 1:
            return torch.sigmoid(logits), edge_preds
        else:
            return logits, edge_preds

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']

            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print(f'[FE-Mamba V3] Encoder weights loaded: {len(new_dict)} keys updated.')

            pretrained_odict = modelCheckpoint['model']
            decoder_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    decoder_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    decoder_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    decoder_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    decoder_dict[new_k] = v

            new_decoder_dict = {k: v for k, v in decoder_dict.items() if k in model_dict.keys()}
            model_dict.update(new_decoder_dict)
            print(f'[FE-Mamba V3] Decoder weights loaded: {len(new_decoder_dict)} keys updated.')

            self.vmunet.load_state_dict(model_dict)
            print(">>> [FE-Mamba V3] Initialized with Aggressive Strategy (Gamma=0.1, EdgeHead=ON)")