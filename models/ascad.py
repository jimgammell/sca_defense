from copy import copy
import numpy as np
import torch
from torch import nn

class MlpBest(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape=(256, 1),
                 layers = 6*[200]):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layers = copy(layers)
        
        layers = [np.prod(input_shape)] + layers
        modules = []
        for li, lo in zip(layers[:-1], layers[1:]):
            modules.append(nn.Linear(li, lo))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(layers[-1], np.prod(output_shape))
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        logits = self.model(x).view(-1, *self.output_shape)
        return logits

class CnnBest(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape=(256, 1),
                 kernel_size=11,
                 input_filters=64):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.kernel_size = kernel_size
        self.input_filters = input_filters
        
        def get_conv_block(in_channels, out_channels, kernel_size, stride, block_name):
            modules = OrderedDict(
                (block_name+'_conv', nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)),
                (block_name+'_relu', nn.ReLU()),
                (block_name+'_pool', nn.AvgPool1d(2))
            )
            block = nn.Sequential(modules)
            return block
        def get_fc_block(input_dims, output_dims, block_name):
            modules = OrderedDict(
                (block_name+'_linear', nn.Linear(input_dims, output_dims)),
                (block_name+'_relu', nn.ReLU())
            )
            block = nn.Sequential(modules)
            return block
        
        self.feature_extractor = nn.Sequential(OrderedDict(
            ('block1', get_conv_block(input_shape[-1], input_filters, kernel_size, stride, 'block1')),
            ('block2', get_conv_block(input_filters, 2*input_filters, kernel_size, stride, 'block2')),
            ('block3', get_conv_block(2*input_filters, 4*input_filters, kernel_size, stride, 'block3')),
            ('block4', get_conv_block(4*input_filters, 8*input_filters, kernel_size, stride, 'block4')),
            ('block5', get_conv_block(8*input_filters, 8*input_filters, kernel_size, stride, 'block5'))
        ))
        eg_input = torch.randn(1, *input_shape)
        self.num_features = np.prod(self.feature_extractor(eg_input).shape[1:])
        self.fc = nn.Sequential(OrderedDict(
            ('block1', get_fc_block(self.num_features, 4096, 'block1')),
            ('block2', get_fc_block(4096, 4096, 'block2')),
            ('block3', get_fc_block(4096, np.prod(output_shape), 'block3'))
        ))
        
    def forward(self, x):
        features = self.feature_extractor(x).view(-1, self.num_features)
        logits = self.fc(features).view(-1, *self.output_shape)
        return logits