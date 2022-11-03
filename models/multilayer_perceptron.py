import numpy as np
import torch
from torch import nn

class MultilayerPerceptron(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 hidden_layer_sizes=[],
                 hidden_activation=nn.ReLU,
                 output_activation=nn.Identity):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        layer_sizes = [np.prod(input_shape)] + hidden_layer_sizes + [np.prod(output_shape)]
        modules = []
        for idx in range(len(layer_sizes)-2):
            modules.append(nn.Linear(layer_sizes[idx], layer_sizes[idx+1]))
            modules.append(hidden_activation())
        modules.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        modules.append(output_activation())
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x.view(-1, np.prod(self.input_shape))).view(-1, *self.output_shape)