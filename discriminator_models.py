import numpy as np
import torch
from torch import nn
from utils import log_print as print

class Discriminator(nn.Module):
    def __init__(self, model, output_transform=None):
        super().__init__()
        self.model = model
        self.output_transform = output_transform
    def forward(self, x):
        output = self.model(x)
        if self.output_transform != None:
            output = self.output_transform(output)
        return output

# Credit: https://github.com/google/scaaml/blob/master/scaaml/intro/model.py
def get_google_style_resnet_discriminator(input_shape,
                                          pool_size=4,
                                          filters=8,
                                          block_kernel_size=3,
                                          activation=nn.ReLU,
                                          dense_dropout=0.1,
                                          num_blocks=[3, 4, 4, 3]):
    class Block(nn.Module):
        def __init__(self,
                     filters,
                     input_filters,
                     kernel_size=3,
                     strides=1,
                     conv_shortcut=False,
                     activation=nn.ReLU):
            super().__init__()
            
            F_modules = [nn.BatchNorm1d(input_filters),
                         activation(),
                         nn.Conv1d(in_channels=input_filters,
                                   out_channels=filters,
                                   kernel_size=1,
                                   bias=False),
                         nn.BatchNorm1d(filters),
                         activation(),
                         nn.Conv1d(in_channels=filters,
                                   out_channels=filters,
                                   kernel_size=kernel_size,
                                   stride=strides,
                                   bias=False,
                                   padding=kernel_size//2),
                         nn.BatchNorm1d(filters),
                         activation(),
                         nn.Conv1d(in_channels=filters,
                                   out_channels=4*filters,
                                   kernel_size=1)]
            self.F = nn.Sequential(*F_modules)

            if conv_shortcut:
                self.Id = nn.Conv1d(in_channels=input_filters,
                                    out_channels=4*filters,
                                    kernel_size=1,
                                    stride=strides)
            else:
                if strides > 1:
                    self.Id = nn.MaxPool1d(kernel_size=1, stride=strides)
                else:
                    self.Id = nn.Identity()
        def forward(self, x):
            return self.F(x) + self.Id(x)
    
    class Stack(nn.Module):
        def __init__(self,
                     filters,
                     input_filters,
                     blocks,
                     kernel_size=3,
                     strides=2,
                     activation=nn.ReLU):
            super().__init__()
            
            modules = [Block(filters,
                             input_filters,
                             kernel_size=kernel_size,
                             activation=activation,
                             conv_shortcut=True)]
            modules.extend([Block(filters,
                                  4*filters,
                                  kernel_size=kernel_size,
                                  activation=activation) for _ in range(2, blocks)])
            modules.append(Block(filters,
                                 4*filters,
                                 strides=strides,
                                 activation=activation))
            self.F = nn.Sequential(*modules)
        def forward(self, x):
            return self.F(x)
            
    modules = [nn.MaxPool1d(kernel_size=4)]
    modules.extend([Stack(filters*2**(block_idx+1),
                          1 if block_idx == 0 else filters*2**(block_idx+2),
                          num_blocks[block_idx],
                          kernel_size=block_kernel_size,
                          activation=activation) for block_idx in range(4)])
    modules.append(nn.Flatten())
    modules.append(nn.LazyLinear(out_features=256))
    model = nn.Sequential(*modules)
    return model
