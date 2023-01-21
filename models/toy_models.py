import numpy as np
import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, n_inputs, n_outputs, bias=True):
        super().__init__()
        
        self.model = nn.Linear(np.prod(n_inputs), n_outputs, bias=bias)
    
    def forward(self, x):
        return self.model(x)