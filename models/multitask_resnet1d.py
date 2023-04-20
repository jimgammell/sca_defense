import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

def apply_sn(modules):
    for idx, module in modules:
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            modules[idx] = spectral_norm(module)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1,
                 activation=lambda: nn.ReLU(inplace=True),
                 use_bn=False, use_sn=False):
        super().__init__()
        
        residual_modules = []
        if use_bn:
            residual_modules.append(nn.BatchNorm1d(in_channels))
        residual_modules.append(activation())
        residual_modules.append(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if use_bn:
            residual_modules.append(nn.BatchNorm1d(out_channels//4))
        residual_modules.append(activation())
        residual_modules.append(
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        )
        if use_bn:
            residual_modules.append(nn.BatchNorm1d(out_channels//4))
        residual_modules.append(activation())
        residual_modules.append(
            nn.Conv1d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0)
        )
        if use_sn:
            apply_sn(residual_modules)
        
        shortcut_modules = []
        if stride != 1:
            shortcut_modules.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
            )
        elif in_channels != out_channels:
            shortcut_modules.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            )
        if use_sn:
            apply_sn(shortcut_modules)
        
        self.residual_connection = nn.Sequential(*residual_modules)
        self.shortcut_connection = nn.Sequential(*shortcut_modules)
    
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.shortcut_connection(x)
        out = x_rc + x_sc
        return out

class Classifier(nn.Module):
    def __init__(self, input_shape, head_sizes,
                 initial_filters=16, block_kernel_size=3,
                 activation=lambda: nn.ReLU(inplace=True), dense_dropout=0.1,
                 num_blocks=[3, 4, 4, 3], use_bn=True, use_sn=False):
        super().__init__()
        
        def get_stack(in_channels, out_channels, blocks,
                      kernel_size=3, stride=2, activation=lambda: nn.ReLU(inplace=True),
                      use_bn=False, use_sn=False):
            modules = []
            modules.append(
                ResidualBlock(in_channels, out_channels, kernel_size=kernel_size,
                              activation=activation, use_sn=use_sn, use_bn=use_bn)
            )
            for _ in range(2, blocks):
                modules.append(
                    ResidualBlock(out_channels, out_channels, kernel_size=kernel_size,
                                  activation=activation, use_sn=use_sn, use_bn=use_bn)
                )
            modules.append(
                ResidualBlock(out_channels, out_channels, kernel_size=kernel_size,
                              activation=activation, stride=stride, use_sn=use_sn, use_bn=use_bn)
            )
            stack = nn.Sequential(*modules)
            return stack
        
        self.input_transform = nn.Conv1d(input_shape[0], initial_filters, kernel_size=3, stride=1, padding=1, bias=False)
        if use_sn:
            self.input_transform = spectral_norm(input_transform)
        
        feature_extractor_modules = []
        filters = initial_filters
        for block_idx in range(4):
            filters *= 2
            feature_extractor_modules.append(
                get_stack(filters//2, filters, num_blocks[block_idx],
                          kernel_size=block_kernel_size, activation=activation,
                          use_bn=use_bn, use_sn=use_sn)
            )
        self.feature_extractor = nn.Sequential(*feature_extractor_modules)
        
        shared_head_modules = []
        shared_head_modules.append(nn.Dropout(dense_dropout))
        shared_head_modules.append(nn.Linear(filters, 256, bias=False))
        if use_bn:
            shared_head_modules.append(nn.BatchNorm1d(256))
        shared_head_modules.append(activation())
        if use_sn:
            apply_sn(shared_head_modules)
        self.shared_head = nn.Sequential(*shared_head_modules)
        
        self.heads = nn.ModuleDict({})
        for head_key, head_size in head_sizes.items():
            if use_sn:
                head = spectral_norm(nn.Linear(256, head_size))
            else:
                head = nn.Linear(256, head_size)
            self.heads.update({str(head_key): head})
    
    def forward(self, x):
        x_i = self.input_transform(x)
        x_fe = self.feature_extractor(x_i)
        features = torch.mean(x_fe, dim=2).view(x_fe.size(0), x_fe.size(1))
        features_sh = self.shared_head(features)
        logits = {head_name: head(features_sh) for head_name, head in self.heads.items()}
        return logits