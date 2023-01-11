import torch
from torch import nn

class BasicWrapper(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
    
    def forward(self, logits, x, y):
        return self.loss_fn.forward(logits, y)
    
    def __repr__(self):
        return 'BasicWrapper({})'.format(self.loss_fn)