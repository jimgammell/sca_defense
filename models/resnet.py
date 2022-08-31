# Code based on https://github.com/google/scaaml/blob/master/scaaml/intro/model.py
# -- Adapted from Keras to PyTorch

import collections
import numpy as np
import torch
from torch import nn

from models.common import get_param_count
from utils import get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))

class Block(nn.Module):
    def __init__(self,
                 eg_input,
                 filters,
                 kernel_size=3,
                 strides=1,
                 conv_shortcut=False,
                 activation=nn.ReLU):
        super().__init__()
        
        self.input_shape = eg_input.shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv_shortcut = conv_shortcut
        self.activation = activation
        
        self.input_transform = nn.Sequential(nn.BatchNorm1d(num_features=eg_input.shape[1],
                                                            momentum=.01,
                                                            eps=1e-3),
                                             activation())
        
        eg_input = self.input_transform(eg_input)
        if conv_shortcut:
            self.shortcut = nn.Conv1d(in_channels=eg_input.shape[1],
                                      out_channels=4*filters,
                                      kernel_size=1,
                                      stride=strides)
        else:
            if strides > 1:
                self.shortcut = nn.MaxPool1d(kernel_size=1,
                                             stride=strides)
            else:
                self.shortcut = nn.Identity()
        
        self.residual = nn.Sequential(nn.Conv1d(in_channels=eg_input.shape[1],
                                                out_channels=filters,
                                                kernel_size=1,
                                                bias=False),
                                      nn.BatchNorm1d(num_features=filters,
                                                     momentum=.01,
                                                     eps=1e-3),
                                      activation(),
                                      nn.ConstantPad1d(padding=(kernel_size//2, kernel_size//2),
                                                       value=0),
                                      nn.Conv1d(in_channels=filters,
                                                out_channels=filters,
                                                kernel_size=kernel_size,
                                                stride=strides,
                                                bias=False),
                                      nn.BatchNorm1d(num_features=filters,
                                                     momentum=.01,
                                                     eps=1e-3),
                                      activation(),
                                      nn.Conv1d(in_channels=filters,
                                                out_channels=4*filters,
                                                kernel_size=1))
        
    def forward(self, x):
        x = self.input_transform(x)
        x = self.shortcut(x) + self.residual(x)
        return x
    
    def __repr__(self):
        s = 'ResNet1D block.' +\
            '\n\tInput shape: {}'.format(self.input_shape) +\
            '\n\tFilters: {}'.format(self.filters) +\
            '\n\tKernel size: {}'.format(self.kernel_size) +\
            '\n\tStrides: {}'.format(self.strides) +\
            '\n\tConvolutional shortcut: {}'.format(self.conv_shortcut) +\
            '\n\tActivation: {}'.format(self.activation) +\
            '\n\tParameter count: {}'.format(get_param_count(self)) +\
            '\nModel summary:' + super(nn.Module, self).__repr__()
        return s

class Stack(nn.Module):
    def __init__(self,
                 eg_input,
                 filters,
                 blocks,
                 kernel_size=3,
                 strides=2,
                 activation=nn.ReLU):
        super().__init__()
        
        self.input_shape = eg_input.shape
        self.filters = filters
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        
        modules = [Block(eg_input,
                         filters,
                         kernel_size=kernel_size,
                         activation=activation,
                         conv_shortcut=True)]
        for _ in range(2, blocks):
            eg_input = modules[-1](eg_input)
            modules.append(Block(eg_input,
                                 filters,
                                 kernel_size=kernel_size,
                                 activation=activation))
        eg_input = modules[-1](eg_input)
        modules.append(Block(eg_input,
                             filters,
                             strides=strides,
                             activation=activation))
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x)
    
    def __repr__(self):
        s = 'ResNet1D stack.' +\
            '\n\tInput shape: {}'.format(self.input_shape) +\
            '\n\tFilters: {}'.format(self.filters) +\
            '\n\tBlocks: {}'.format(self.blocks) +\
            '\n\tKernel size: {}'.format(self.kernel_size) +\
            '\n\tStrides: {}'.format(self.strides) +\
            '\n\tActivation: {}'.format(self.activation) +\
            '\n\tParameter count: {}'.format(get_param_count(self)) +\
            '\nModel summary:\n' + super(nn.Module, self).__repr__()
        return s

class ResNet1D(nn.Module):
    def __init__(self,
                 eg_input_shape,
                 pool_size=4,
                 filters=8,
                 block_kernel_size=3,
                 activation=nn.ReLU,
                 dense_dropout=0.1,
                 num_blocks=[3, 4, 4, 3]):
        super().__init__()
        
        self.input_shape = eg_input_shape
        self.pool_size = pool_size
        self.filters = filters
        self.block_kernel_size = block_kernel_size
        self.activation = activation
        self.dense_dropout = dense_dropout
        self.num_blocks = num_blocks
        
        eg_input = torch.rand(eg_input_shape)
        self.input_transform = nn.Sequential(nn.MaxPool1d(kernel_size=pool_size))
        eg_input = self.input_transform(eg_input)
        
        modules = []
        for block_idx in range(4):
            filters *= 2
            modules.append(Stack(eg_input,
                                 filters,
                                 num_blocks[block_idx],
                                 kernel_size=block_kernel_size,
                                 activation=activation))
            eg_input = modules[-1](eg_input)
        self.feature_extractor = nn.Sequential(*modules)
        
        self.feature_reducer = nn.Sequential(nn.AvgPool1d(kernel_size=eg_input.shape[-1]),
                                             nn.Flatten(1, -1))
        eg_input = self.feature_reducer(eg_input)
        
        self.dense_probe = nn.Sequential(nn.Dropout(dense_dropout),
                                         nn.Linear(eg_input.shape[1], 256),
                                         nn.BatchNorm1d(num_features=256,
                                                        momentum=.01,
                                                        eps=1e-3),
                                         activation(),
                                         nn.Linear(256, 256))
        eg_input = self.dense_probe(eg_input)
        
    def forward(self, x):
        x = self.input_transform(x)
        x = self.feature_extractor(x)
        x = self.feature_reducer(x)
        x = self.dense_probe(x)
        return x
    
    def __repr__(self):
        s = 'ResNet1D model.' +\
            '\n\tInput shape: {}'.format(self.input_shape) +\
            '\n\tPool size: {}'.format(self.pool_size) +\
            '\n\tFilters: {}'.format(self.filters) +\
            '\n\tBlock kernel size: {}'.format(self.block_kernel_size) +\
            '\n\tActivation: {}'.format(self.activation) +\
            '\n\tDense dropout: {}'.format(self.dense_dropout) +\
            '\n\tNumber of blocks: {}'.format(self.num_blocks) +\
            '\n\tParameter count: {}'.format(get_param_count(self)) +\
            '\nModel summary:\n' + super(nn.Module, self).__repr__()
        return s