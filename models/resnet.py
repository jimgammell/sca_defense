# This is an adaptation from Keras to PyTorch of the following file (not my own):
#  https://github.com/google/scaaml/blob/master/scaaml/intro/model.py

import numpy as np
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 kernel_size=3,
                 strides=1,
                 conv_shortcut=False,
                 activation=nn.ReLU):
        super().__init__()
        
        self.preprocess = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            activation()
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, 4*filters, 1, stride=strides) if conv_shortcut else
            nn.MaxPool1d(1, stride=strides) if strides > 1 else
            nn.Identity()
        )
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, filters, 1, bias=False, padding=1//2),
            nn.BatchNorm1d(filters),
            activation(),
            nn.Conv1d(filters, filters, kernel_size, stride=strides, bias=False, padding=kernel_size//2),
            nn.BatchNorm1d(filters),
            activation(),
            nn.Conv1d(filters, 4*filters, 1)
        )
    
    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv_block(x) + self.shortcut(x)
        return x

class Stack(nn.Module):
    def __init__(self,
                 in_channels,
                 filters,
                 blocks,
                 kernel_size=3,
                 strides=2,
                 activation=nn.ReLU):
        super().__init__()
        
        self.stack = nn.Sequential(
            Block(in_channels, filters, kernel_size=kernel_size, activation=activation, conv_shortcut=True),
            *[Block(4*filters, filters, kernel_size=kernel_size, activation=activation) for _ in range(2, blocks)],
            Block(4*filters, filters, strides=strides, activation=activation)
        )
        
    def forward(self, x):
        x = self.stack(x)
        return x

class ResNet1D(nn.Module):
    def __init__(self,
                 input_shape,
                 pool_size=1, # Initial pooling moved to the dataset code to avoid keeping a whole trace in ram only to downsample it
                 filters=8,
                 block_kernel_size=3,
                 activation=nn.ReLU,
                 dense_dropout=0.1,
                 num_blocks=[3, 4, 4, 3]):
        super().__init__()
        
        self.preprocess = nn.Sequential(
            nn.MaxPool1d(pool_size)
        )
        self.trunk = nn.Sequential(
          *[Stack(1 if l == 0 else 4*filters*2**l,
                  filters*2**(l+1),
                  num_blocks[l],
                  kernel_size=block_kernel_size,
                  activation=activation)
            for l in range(4)]
        )
        features_shape = self.trunk(self.preprocess(torch.randn(1, *input_shape))).shape
        self.head = nn.Sequential(
            nn.AvgPool1d(features_shape[-1]),
            nn.Flatten(),
            nn.Dropout(dense_dropout),
            nn.Linear(features_shape[-2], 256),
            nn.BatchNorm1d(256),
            activation(),
            nn.Linear(256, 256)
        ) # softmax layer omitted
        
    def forward(self, x):
        x = self.preprocess(x)
        x = self.trunk(x)
        x = self.head(x)
        return x