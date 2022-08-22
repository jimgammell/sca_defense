import numpy as np
from torch import nn

class MultilayerPerceptron(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 hidden_layers=[],
                 hidden_activation=nn.ReLU):
        super().__init__()
        modules = []
        layers = [n_inputs] + hidden_layers + [n_outputs]
        for l1, l2 in zip(layers[:-2], layers[1:-1]):
            modules.append(nn.Linear(l1, l2))
            modules.append(hidden_activation())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x)