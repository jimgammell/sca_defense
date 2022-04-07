import numpy as np
import torch
from torch import nn

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
                     kernel_size=3,
                     strides=1,
                     conv_shortcut=False,
                     activation=nn.ReLU):
            super().__init__()
            
            F_modules = [nn.LazyBatchNorm1d(),
                         activation(),
                         nn.LazyConv1d(out_channels=filters,
                                       kernel_size=1,
                                       use_bias=False,
                                       padding='same'),
                         nn.LazyBatchNorm1d(),
                         activation(),
                         nn.LazyConv1d(out_channels=filters,
                                       kernel_size=kernel_size,
                                       strides=strides,
                                       use_bias=False,
                                       padding='same'),
                         nn.LazyBatchNorm1D(),
                         activation(),
                         nn.LazyConv1d(out_channels=4*filters,
                                       kernel_size=1)]
            self.F = nn.Sequential(*F_modules)

            if conv_shortcut:
                self.Id = nn.LazyConv1d(out_channels=4*filters,
                                        kernel_size=1,
                                        stride=strides)
            else:
                if strides > 1:
                    self.Id = nn.MaxPooling1d(pool_size=1, strides=strides)
                else:
                    self.Id = nn.Identity()
        def forward(self, x):
            return self.F(x) + self.Id(x)
    
    class Stack(nn.Module):
        def __init__(self,
                     filters,
                     blocks,
                     kernel_size=3,
                     strides=2,
                     activation=nn.ReLU):
            super().__init__()
            
            modules = [Block(filters,
                             kernel_size=kernel_size,
                             activation=activation,
                             conv_shortcut=True)]
            modules.extend([Block(filters,
                                  kernel_size=kernel_size,
                                  activation=activation) for _ in range(2, blocks)])
            modules.append(Block(filters,
                                 strides=strides,
                                 activation=activation))
            self.F = nn.Sequential(*modules)
        def forward(self, x):
            return self.F(x)
            
    modules = [nn.MaxPool1d(pool_size=4)]
    modules.extend([Stack(filters*2**(block_idx+1),
                          num_blocks[block_idx],
                          kernel_size=block_kernel_size,
                          activation=activation) for block_idx in range(4)])
    modules.append(nn.LazyLinear(out_features=256))
    model = nn.Sequential(*modules)
    return model
