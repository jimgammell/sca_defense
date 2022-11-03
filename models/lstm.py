import numpy as np
import torch
from torch import nn

class LstmModel(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 delay=1,
                 dropout=0.):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.delay = delay
        self.recurrent_model = nn.LSTM(input_size=1,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       batch_first=True,
                                       dropout=dropout)
        self.output_transform = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        batch_size, sequence_length = x.size(0), x.size(1)
        x = x.view(batch_size, sequence_length, 1)
        if self.delay != 0:
            x = x[:, :-self.delay, :]
            delay_padding = torch.zeros(batch_size, self.delay, 1).to(x.get_device())
            x = torch.cat((delay_padding, x), dim=1)
        output, _ = self.recurrent_model(x)
        output = self.output_transform(output).view(batch_size, sequence_length)
        return output