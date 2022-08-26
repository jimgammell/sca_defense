import numpy as np
from torch import nn

from utils import get_print_to_log
print = get_print_to_log(__file__)

class MultilayerPerceptron(nn.Module):
    def __init__(self, n_inputs, n_outputs,
                 hidden_layers=[],
                 hidden_activation=nn.ReLU,
                 batch_norm=False,
                 dropout=0.):
        super().__init__()
        modules = []
        layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
        if type(hidden_activation) == list:
            assert len(hidden_activation) == len(hidden_layers)
        else:
            hidden_activation = len(hidden_layers)*[hidden_activation]
        if batch_norm == False:
            batch_norm = (len(layer_sizes)-1)*[False]
        elif batch_norm == True:
            batch_norm = (len(layer_sizes)-1)*[True]
        else:
            assert len(batch_norm) == len(layer_sizes)-1
        if type(dropout) == float:
            dropout = (len(layer_sizes)-1)*[dropout]
        else:
            assert len(dropout) == len(layer_sizes)-1
        if batch_norm[0] != False:
            modules.append(nn.BatchNorm1d(layer_sizes[0]))
        if dropout[0] != 0.:
            modules.append(nn.Dropout(dropout[0]))
        for idx in range(len(layer_sizes)-2):
            modules.append(nn.Linear(layer_sizes[idx], layer_sizes[idx+1]))
            if batch_norm[idx+1] != False:
                modules.append(nn.BatchNorm1d(layer_sizes[idx+1]))
            modules.append(hidden_activation[idx]())
            if dropout[idx+1] != 0.:
                modules.append(nn.Dropout(dropout[idx+1]))
        modules.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.model = nn.Sequential(*modules)
        
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.dropout = dropout
        
        
    def forward(self, x):
        return self.model(x)
    
    def __repr__(self):
        s = 'Multilayer Perceptron model:' +\
            '\n\tLayer sizes: %s'%(', '.join(str(x) for x in self.layer_sizes)) +\
            '\n\tHidden activations: %s'%(', '.join(str(h) for h in self.hidden_activation)) +\
            '\n\tBatch norm: %s'%(', '.join(str(b) for b in self.batch_norm)) +\
            '\n\tDropout: %s'%(', '.join(str(d) for d in elf.dropout))
        return s
    
    def summary(self):
        print(super().__repr__())

class Linear(MultilayerPerceptron):
    def __init__(self, n_inputs, n_outputs):
        super().__init__(n_inputs, n_outputs)
        
    def __repr__(self):
        s = 'Linear model:' +\
            '\n\tLayer sizes: %s'%(', '.join(str(x) for x in self.layer_sizes))
        return s
    
    def summary(self):
        print(self.model.__repr__())

class XDeepSca(MultilayerPerceptron):
    def __init__(self, n_inputs, n_outputs,
                 hidden_layers=[200, 200],
                 hidden_activation=nn.ReLU,
                 batch_norm=[False, True, True],
                 dropout=[0., .1, .05]):
        super().__init__(n_inputs, n_outputs,
                         hidden_layers=hidden_layers,
                         hidden_activation=hidden_activation,
                         batch_norm=batch_norm,
                         dropout=dropout)
    
    def __repr__(self):
        s = 'X-DeepSCA model:' +\
            '\n\tLayer sizes: %s'%(', '.join(str(x) for x in self.layer_sizes)) +\
            '\n\tHidden activations: %s'%(', '.join(str(a) for a in self.hidden_activation)) +\
            '\n\tBatch norm: %s'%(', '.join(str(b) for b in self.batch_norm)) +\
            '\n\tDropout: %s'%(', '.join(str(d) for d in self.dropout))
        return s
    
    def summary(self):
        print(self.model.__repr__())