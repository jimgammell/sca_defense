import numpy as np
import torch
from torch import nn

class ZeroModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        n_inputs = np.prod(n_inputs)
        n_outputs = np.prod(n_outputs)
        self.model = nn.Linear(n_inputs, n_outputs)
    
    def forward(self, x):
        return 0*self.model(x)

class LinearModel(nn.Module):
    def __init__(self, n_inputs, n_outputs, bottleneck_width=None, bias=True, sn=False):
        super().__init__()
        
        n_inputs = int(np.prod(n_inputs))
        n_outputs = int(np.prod(n_outputs))
        modules = []
        if bottleneck_width is not None:
            modules.append(nn.Linear(n_inputs, bottleneck_width, bias=bias))
            modules.append(nn.Linear(bottleneck_width, n_outputs, bias=bias))
        else:
            modules.append(nn.Linear(n_inputs, n_outputs, bias=bias))
        if sn:
            modules[-1] = nn.utils.spectral_norm(modules[-1])
        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.model(x)

class MlpModel(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_width, hidden_depth, hidden_activation=nn.ReLU):
        super().__init__()
        
        n_inputs = int(np.prod(n_inputs))
        n_outputs = int(np.prod(n_outputs))
        assert hidden_depth >= 0
        if hidden_depth == 0:
            self.model = nn.Linear(n_inputs, n_outputs)
        else:
            modules = [nn.Linear(n_inputs, hidden_width), hidden_activation()]
            for _ in range(hidden_depth-1):
                modules.extend([nn.Linear(hidden_width, hidden_width), hidden_activation()])
            modules.append(nn.Linear(hidden_width, n_outputs))
            self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x)