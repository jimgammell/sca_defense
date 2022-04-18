import numpy as np
import torch
from torch import nn
import sys
thismod = sys.modules[__name__]

def get_loss_constructor(name):
    try:
        import sys
        thismod = sys.modules[__name__]
        loss_constructor = getattr(thismod, name)
    except:
        loss_constructor = getattr(nn, name)
    return loss_constructor

class MseFromLabel:
    def __init__(self):
        self.base_loss_fn = nn.MSELoss()
        self.to_oh_fn = lambda x: nn.functional.one_hot(x, num_classes=256).type(torch.float)
    def __call__(self, logits, target):
        return self.base_loss_fn(logits, self.to_oh_fn(target))

class NegativeLoss:
    def __init__(self, loss_constructor, l2_coeff=0, **loss_constructor_kwargs):
        loss_constructor = get_loss_constructor(loss_constructor)
        base_loss_fn = loss_constructor(**loss_constructor_kwargs)
        self.loss_fn = lambda x, y: -base_loss_fn(x, y)
        self.l2_coeff = l2_coeff
    def __call__(self, logits, target, gen_output):
        if self.l2_coeff != 0:
            l2_loss = self.l2_coeff*torch.std(gen_output)
        else:
            l2_loss = 0
        loss = self.loss_fn(logits, target)
        return loss + l2_loss

class NormLoss:
    def __init__(self, p):
        if p == 'inf':
            p = np.inf
        self.norm_function = lambda x: torch.linalg.norm(x, ord=p, dim=-1)
    def __call__(self, logits, target):
        prediction = torch.nn.functional.softmax(logits, dim=-1)
        elementwise_loss = self.norm_function(prediction)
        loss = torch.mean(elementwise_loss)
        return loss

class BatchStdLoss:
    def __init__(self):
        pass
    def __call__(self, logits, target):
        mean = torch.mean(logits, dim=0)
        elementwise_loss = torch.linalg.norm(logits-mean, ord=2, dim=0)
        loss = torch.mean(elementwise_loss)
        return loss