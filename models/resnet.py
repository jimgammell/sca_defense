# Code based on https://github.com/google/scaaml/blob/master/scaaml/intro/model.py
# -- Adapted from Keras to PyTorch

import collections
import numpy as np
import torch
from torch import nn

from utils import get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))

class Block(nn.Module):
    def __init__(self,
                 input_filters,
                 output_filters,
                 kernel_size=3,
                 strides=1,
                 conv_shortcut=False,
                 activation=nn.ReLU):
        super().__init__()
        
        self.input_transform = nn.Sequential(nn.BatchNorm1d(input_filters),
                                             activation())
        
        if conv_shortcut:
            self.shortcut = nn.Conv1d(in_channels=input_filters,
                                      out_channels=4*output_filters,
                                      kernel_size=1,
                                      stride=strides)
        else:
            if strides > 1:
                self.shortcut = nn.Sequential(nn.ConstantPad1d((1, 1), -np.inf),
                                              nn.MaxPool1d(1, stride=strides))
            else:
                self.shortcut = nn.Identity()
                
        self.residual = nn.Sequential(nn.Conv1d(input_filters,
                                                output_filters,
                                                1,
                                                bias=False),
                                      nn.BatchNorm1d(output_filters),
                                      activation(),
                                      nn.Conv1d(output_filters,
                                                output_filters,
                                                kernel_size,
                                                stride=strides,
                                                bias=False),
                                      nn.ConstantPad1d(kernel_size//2, 0),
                                      nn.BatchNorm1d(output_filters),
                                      activation(),
                                      nn.Conv1d(output_filters,
                                                4*output_filters,
                                                1))
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv_shortcut = conv_shortcut
        self.activation = activation
        
        
    def forward(self, x):
        x = self.input_transform(x)
        x = self.shortcut(x) + self.residual(x)
        return x
    
    def __repr__(self):
        s = 'ResNet block:' +\
            '\n\tInput filters: %d'%(self.input_filters) +\
            '\n\tOutput filters: %d'%(self.output_filters) +\
            '\n\tKernel size: %d'%(self.kernel_size) +\
            '\n\tStrides: %d'%(self.strides) +\
            '\n\tConvolutional shortcut: {}'.format(self.conv_shortcut) +\
            '\n\tActivation: {}'.format(self.activation)
        return s
    
    def summary(self):
        print(super().__repr__())

class Stack(nn.Module):
    def __init__(self,
                 input_filters,
                 output_filters,
                 blocks,
                 kernel_size=3,
                 strides=2,
                 activation=nn.ReLU):
        super().__init__()
        
        modules = [Block(input_filters,
                         output_filters,
                         kernel_size=kernel_size,
                         activation=activation,
                         conv_shortcut=True)]
        for _ in range(2, blocks):
            modules.append(Block(4*output_filters,
                                 output_filters,
                                 kernel_size=kernel_size,
                                 activation=activation))
        modules.append(Block(4*output_filters,
                             output_filters,
                             strides=strides,
                             activation=activation))
        self.modules = modules
        self.stack = nn.Sequential(*modules)
        
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.blocks = blocks
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        
    def forward(self, x):
        return self.stack(x)
    
    def __repr__(self):
        s = 'ResNet stack:' +\
            '\n\tInput filters: %d'%(self.input_filters) +\
            '\n\tOutput filters: %d'%(self.output_filters) +\
            '\n\tBlocks: %d'%(self.blocks) +\
            '\n\tKernel size: %d'%(self.kernel_size) +\
            '\n\tActivation: {}'.format(self.activation)
        for block_idx, block in enumerate(self.modules):
            print('\t\nBlock %d:'%(block_idx))
            print(block)
        return s
    
    def summary(self):
        print(super().__repr__())

class ResNet1D(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 pool_size=4,
                 filters=8,
                 block_kernel_size=3,
                 activation=nn.ReLU,
                 dense_dropout=0.1,
                 num_blocks=[3, 4, 4, 3]):
        super().__init__()
        
        modules = [('input_reshaping', nn.Unflatten(1, (1, n_inputs))),
                   ('input_downsampling', nn.MaxPool1d(pool_size))]
        input_filters = 1
        output_filters = filters
        for block_idx in range(len(num_blocks)):
            output_filters *= 2
            modules.append(('stack_%d'%(block_idx), Stack(input_filters,
                                                          output_filters,
                                                          num_blocks[block_idx],
                                                          kernel_size=block_kernel_size,
                                                          activation=activation)))
            input_filters = 4*output_filters
        eg_input = torch.rand((64, n_inputs))
        for _, module in modules:
            eg_input = module(eg_input)
        modules.append(('feature_reducer', nn.Sequential(nn.AvgPool1d(eg_input.shape[-1]),
                                                         nn.Flatten(1, -1))))
        
        modules.append(('dense_probe', nn.Sequential(nn.Dropout(dense_dropout),
                                                     nn.Linear(input_filters, n_outputs),
                                                     nn.BatchNorm1d(n_outputs),
                                                     activation(),
                                                     nn.Linear(n_outputs, n_outputs))))
        self.model = nn.Sequential(collections.OrderedDict(modules))
        
        self.input_size = n_inputs
        self.output_size = n_outputs
        self.pool_size = pool_size
        self.filters = filters
        self.block_kernel_size = block_kernel_size
        self.activation = activation
        self.dense_dropout = dense_dropout
        self.num_blocks = num_blocks
    
    def forward(self, x):
        return self.model(x)
    
    def __repr__(self):
        s = 'ResNet1d:' +\
            '\n\tInput size: %d'%(self.input_size) +\
            '\n\tOutput size: %d'%(self.output_size) +\
            '\n\tPool size: %d'%(self.pool_size) +\
            '\n\tFilters: %d'%(self.filters) +\
            '\n\tBlock kernel size: %d'%(self.block_kernel_size) +\
            '\n\tActivation: {}'.format(self.activation) +\
            '\n\tDense dropout: %.01f'%(self.dense_dropout) +\
            '\n\tBlock sizes: %s'%(', '.join(str(b) for b in self.num_blocks))
        return s
    
    def summary(self):
        print(super().__repr__())