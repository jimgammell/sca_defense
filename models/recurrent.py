import numpy as np
import torch
from torch import nn

class LstmModel(nn.Module):
    def __init__(self, input_shape, layers=[64, 64, 64], delay=1, samples_per_timestep=1, realistic_hindsight=True):
        super().__init__()
        
        if len(layers) > 0:
            self.recurrent_layers = nn.ModuleList([nn.LSTMCell(samples_per_timestep, layers[0])])
            for li, lo in zip(layers[:-1], layers[1:]):
                self.recurrent_layers.append(nn.LSTMCell(li, lo))
            self.recurrent_layers.append(nn.LSTMCell(layers[-1], samples_per_timestep))
        else:
            self.recurrent_layers = nn.ModuleList([nn.LSTMCell(samples_per_timestep, samples_per_timestep)])
        
        self.input_shape = input_shape
        self.layers = layers
        self.delay = delay
        self.samples_per_timestep = samples_per_timestep
        self.realistic_hindsight = realistic_hindsight
    
    def forward(self, x):
        x = x.transpose(-1, -2)
        if self.delay != 0:
            delay_padding = torch.zeros_like(x[:, :self.delay, :])
            x = torch.cat((delay_padding, x[:, :-self.delay, :]), dim=1)
        x = x.reshape(-1, x.size(1)//self.samples_per_timestep, self.samples_per_timestep)
        h = [torch.zeros(x.size(0), l, device=x.device) for l in self.layers+[self.samples_per_timestep]]
        c = [torch.zeros(x.size(0), l, device=x.device) for l in self.layers+[self.samples_per_timestep]]
        mask = torch.zeros_like(x)
        for t, x_t in enumerate(x.split(1, dim=1)):
            x_t = x_t.squeeze(1)
            if self.realistic_hindsight and t > 0:
                x_t = x_t - mask[:, t-1, :]
            h[0], c[0] = self.recurrent_layers[0](x_t, (h[0], c[0]))
            for idx in range(len(self.recurrent_layers[1:])):
                h[idx+1], c[idx+1] = self.recurrent_layers[idx+1](h[idx], (h[idx+1], c[idx+1]))
            mask[:, t, :] = h[-1]
        mask = mask.reshape(mask.size(0), -1, 1).transpose(-1, -2)
        return mask