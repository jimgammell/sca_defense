import numpy as np
import torch

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