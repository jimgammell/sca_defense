import numpy as np
import torch
from torch import nn

class LstmModel(nn.Module):
    def __init__(self, layers=[64, 64, 64], delay=1, io_dims=1, realistic_hindsight=False):
        super().__init__()
        
        self.layers = layers
        self.delay = delay
        self.io_dims = io_dims
        self.realistic_hindsight = realistic_hindsight
        
        self.recurrent_layers = nn.ModuleList([nn.LSTMCell(io_dims, layers[0])])
        for li, lo in zip(layers[:-1], layers[1:]):
            self.recurrent_layers.append(nn.LSTMCell(li, lo))
        self.recurrent_layers.append(nn.Linear(layers[-1], io_dims))
        
    def forward(self, x):
        x = x.clone()
        if self.delay != 0:
            delay_padding = torch.zeros_like(x[:, :self.delay, :])
            x = torch.cat((delay_padding, x[:, :-self.delay, :]), dim=1)
        h = [torch.zeros(x.size(0), l, device=x.device) for l in self.layers]
        c = [torch.zeros(x.size(0), l, device=x.device) for l in self.layers]
        mask = []
        for x_t in x.split(1, dim=1):
            x_t = x_t.squeeze(1)
            if self.realistic_hindsight and len(mask) > 0:
                x_t = x_t - mask[-1]
            h[0], c[0] = self.recurrent_layers[0](x_t, (h[0], c[0]))
            for idx in range(len(self.recurrent_layers[1:-1])):
                h[idx+1], c[idx+1] = self.recurrent_layers[idx+1](h[idx], (h[idx+1], c[idx+1]))
            o_t = self.recurrent_layers[-1](h[-1])
            mask.append(o_t)
        mask = torch.cat(mask, dim=1).view(x.shape)
        return mask