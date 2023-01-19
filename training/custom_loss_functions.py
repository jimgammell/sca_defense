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

class AutoencoderMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_fn = nn.MSELoss()
    
    def forward(self, logits, x, y):
        return self.mse_loss_fn.forward(logits, x)
    
    def __repr__(self):
        return 'AutoencoderMSELoss()'