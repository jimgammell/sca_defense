import numpy as np
import torch
from torch import nn

class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.tensor(x).to(torch.float)
class ToLabelTensor(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.tensor(x).to(torch.long)
    
class RandomNoise(nn.Module):
    def __init__(self, max_convex_coef=0.05):
        super().__init__()
        self.max_convex_coef = max_convex_coef
    def forward(self, x):
        convex_coeffs = self.max_convex_coef*torch.rand(x.shape[-1], device=x.device)
        noise = torch.randn_like(x)
        return noise*convex_coeffs + x*(1-convex_coeffs)