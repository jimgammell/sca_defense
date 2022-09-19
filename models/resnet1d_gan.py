import numpy as np
import torch
from torch import nn

# Batch norm with the default Tensorflow arguments.
def BatchNorm1d(num_features):
    return nn.BatchNorm1d(num_features=num_features,
                          momentum=0.01,
                          eps=1e-3)

class ResNet1dDiscriminator(nn.Module):
    class Block(nn.Module):
        def __init__(self,
                     eg_input,
                     filters,
                     kernel_size=3,
                     strides=1,
                     conv_shortcut=False,
                     activation=nn.LeakyReLU,
                     activation_kwargs={'negative_slope': 0.2},
                     spectral_norm=False):
            super().__init__()
            
            self.input_shape = eg_input.shape
            self.filters = filters
            self.kernel_size = kernel_size
            self.strides = strides
            self.conv_shortcut = conv_shortcut
            self.activation = activation
            self.activation_kwargs = activation_kwargs
            self.spectral_norm = spectral_norm
            def Conv1d(*args, **kwargs):
                if self.spectral_norm:
                    return torch.nn.utils.spectral_norm(nn.Conv1d(*args, **kwargs))
                else:
                    return nn.Conv1d(*args, **kwargs)
            def Activation():
                return self.activation(**self.activation_kwargs)
            
            self.input_transform = nn.Sequential(
                BatchNorm1d(num_features=eg_input.shape[1]),
                Activation())
            eg_input = self.input_transform(eg_input)
            
            if self.conv_shortcut:
                self.shortcut = Conv1d(in_channels=eg_input.shape[1],
                                       out_channels=4*filters,
                                       kernel_size=1,
                                       stride=strides)
            else:
                if strides > 1:
                    self.shortcut = nn.MaxPool1d(kernel_size=1,
                                                 stride=strides)
                else:
                    self.shortcut = nn.Identity()
            
            self.residual = nn.Sequential(
                Conv1d(in_channels=eg_input.shape[1],
                       out_channels=filters,
                       kernel_size=1,
                       bias=False),
                BatchNorm1d(num_features=filters),
                Activation(),
                nn.ConstantPad1d(padding=(kernel_size//2, kernel_size//2),
                                 value=0),
                Conv1d(in_channels=filters,
                       out_channels=filters,
                       kernel_size=kernel_size,
                       stride=strides,
                       bias=False),
                BatchNorm1d(num_features=filters),
                Activation(),
                Conv1d(in_channels=filters,
                       out_channels=4*filters,
                       kernel_size=1))
        
        def forward(self, x):
            transformed_x = self.input_transform(x)
            id_x = self.shortcut(transformed_x)
            resid_x = self.residual(transformed_x)
            logits = id_x + resid_x
            return logits
    
    class Stack(nn.Module):
        def __init__(self,
                     eg_input,
                     filters,
                     blocks,
                     kernel_size=3,
                     strides=2,
                     activation=nn.LeakyReLU,
                     activation_kwargs={'negative_slope': 0.2},
                     spectral_norm=False):
            super().__init__()
            
            self.input_shape = eg_input.shape
            self.filters = filters
            self.blocks = blocks
            self.kernel_size = kernel_size
            self.strides = strides
            self.activation = activation
            self.activation_kwargs = activation_kwargs
            self.spectral_norm = spectral_norm
            def Block(eg_input, kernel_size=None, strides=None, conv_shortcut=None):
                kwargs = {'filters': self.filters,
                          'activation': self.activation,
                          'activation_kwargs': self.activation_kwargs,
                          'spectral_norm': self.spectral_norm}
                if kernel_size is not None:
                    kwargs.update({'kernel_size': kernel_size})
                if strides is not None:
                    kwargs.update({'strides': strides})
                if conv_shortcut is not None:
                    kwargs.update({'conv_shortcut': conv_shortcut})
                return self.Block(eg_input, **kwargs)
            
            modules = [Block(eg_input,
                             kernel_size=self.kernel_size,
                             conv_shortcut=self.conv_shortcut)]
            for _ in range(2, self.blocks):
                eg_input = modules[-1](eg_input)
                modules.append(Block(eg_input,
                                     kernel_size=self.kernel_size))
            eg_input = modules[-1](eg_input)
            modules.append(Block(eg_input,
                                 strides=strides))
            self.model = nn.Sequential(*modules)
            
        def forward(self, x):
            logits = self.model(x)
            return logits
    
    def __init__(self,
                 input_shape,
                 pool_size=4,
                 filters=8,
                 block_kernel_size=3,
                 activation=nn.LeakyReLU,
                 activation_kwargs={'negative_slope': 0.2},
                 spectral_norm=False,
                 dense_dropout=0.1,
                 stack_sizes=[3, 4, 4, 3]):
        super().__init__()
        
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.filters = filters
        self.block_kernel_size = block_kernel_size
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        self.spectral_norm = spectral_norm
        self.dense_dropout = dense_dropout
        self.stack_sizes = stack_sizes
        
        eg_input = torch.randn(self.input_shape)
        self.input_transform = nn.MaxPool1d(kernel_size=pool_size)
        eg_input = self.input_transform(eg_input)
        
        modules = []
        for stack_idx, stack_size in enumerate(stack_sizes):
            filters *= 2
            modules.append(self.Stack(eg_input,
                                      filters,
                                      stack_size,
                                      kernel_size=block_kernel_size,
                                      activation=self.activation,
                                      activation_kwargs=self.activation_kwargs,
                                      spectral_norm=self.spectral_norm))
            eg_input = modules[-1](eg_input)
        self.feature_extractor = nn.Sequential(*modules)
        
        self.pooling_layer = nn.AvgPool1d(kernel_size=eg_input.shape[1])
        eg_input = self.pooling_layer(eg_input)
        eg_input = eg_input.view(-1, np.prod(eg_input.shape[1:]))
        
        self.dense_probe = nn.Sequential(
            nn.Dropout(self.dense_dropout),
            nn.Linear(eg_input.shape[1], 256),
            BatchNorm1d(num_features=256),
            self.activation(**self.activation_kwargs),
            nn.Linear(256, 256))
        eg_input = self.dense_probe(eg_input)
        
    def forward(self, x):
        transformed_x = self.input_transform(x)
        features = self.feature_extractor(transformed_x)
        pooled_features = self.pooling_layer(features)
        logits = self.dense_probe(pooled_features.view(-1, np.prod(pooled_features.shape[1:])))
        return logits