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

class AdversarialEntropyMaximizationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss_fn = nn.BCELoss()
        
    def forward(self, disc_logits, gen_logits, x, y):
        disc_dist = nn.functional.softmax(disc_logits, dim=-1)
        maximum_entropy_dist = nn.functional.softmax(torch.zeros_like(disc_logits), dim=-1)
        loss = self.bce_loss_fn(disc_dist, maximum_entropy_dist)
        return loss
    
    def __repr__(self):
        return self.__class__.__name__+'()'

class AdversarialMinimaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, disc_logits, gen_logits, x, y):
        disc_loss = self.ce_loss_fn(disc_logits, y)
        loss = -disc_loss
        return loss
    
    def __repr__(self):
        return self.__class__.__name__+'()'