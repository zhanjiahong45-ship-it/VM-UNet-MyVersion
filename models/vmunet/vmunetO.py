from .vmambaO import VSSM
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
        logits = self.vmunet(x)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            print(f"Loading weights from {self.load_ckpt_path}...")
            model_dict = self.vmunet.state_dict()
            try:
                modelCheckpoint = torch.load(self.load_ckpt_path, map_location='cpu')
            except FileNotFoundError:
                print(f"Error: Checkpoint not found at {self.load_ckpt_path}")
                return

            pretrained_dict = modelCheckpoint['model'] if 'model' in modelCheckpoint else modelCheckpoint

            new_dict = {}
            for k, v in pretrained_dict.items():
                if k not in model_dict:
                    continue

                # Case 1: Shape Match (Load directly)
                if v.shape == model_dict[k].shape:
                    new_dict[k] = v

                # Case 2: Shape Mismatch (4 -> 8 Directions) - Smart Cloning
                elif v.shape != model_dict[k].shape and (v.shape[0] == 4 and model_dict[k].shape[0] == 8):
                    print(f"Smart Loading {k}: Cloning 4-dir weights to 8-dir...")
                    # Clone the weights: [4, ...] -> [8, ...]
                    dim0_extended = torch.cat([v, v], dim=0)

                    if dim0_extended.shape == model_dict[k].shape:
                        new_dict[k] = dim0_extended
                    else:
                        print(
                            f"WARNING: Shape mismatch after extension for {k}. Expected {model_dict[k].shape}, got {dim0_extended.shape}. Skipping.")

            model_dict.update(new_dict)

            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(
                len(model_dict), len(pretrained_dict), len(new_dict)))

            self.vmunet.load_state_dict(model_dict)
            print("Weights Loaded! (Standard layers: Loaded | Spiral layers: Cloned from Standard)")