import numpy as np
import torch
from torch import nn

class EffNetDS50(nn.Module):
    def __init__(self, input_shape, output_shape, width_multiplier=1, dropout=0., input_dropout=0.):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Dropout(input_dropout),
            nn.Conv1d(input_shape[0], 32*width_multiplier, 1, padding=0),
            nn.SELU(),
            nn.BatchNorm1d(32*width_multiplier),
            nn.AvgPool1d(2),
            nn.Conv1d(32*width_multiplier, 64*width_multiplier, 25, padding=12),
            nn.SELU(),
            nn.BatchNorm1d(64*width_multiplier),
            nn.AvgPool1d(25),
            nn.Conv1d(64*width_multiplier, 128*width_multiplier, 3, padding=1),
            nn.SELU(),
            nn.BatchNorm1d(128*width_multiplier),
            nn.AvgPool1d(4))
        eg_input = torch.randn(1, *input_shape)
        eg_input = self.feature_extractor(eg_input)
        self.num_features = np.prod(eg_input.shape)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 15*width_multiplier),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(15*width_multiplier, 15*width_multiplier),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(15*width_multiplier, 15*width_multiplier),
            nn.SELU(),
            nn.Dropout(dropout),
            nn.Linear(15*width_multiplier, np.prod(output_shape)))
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_features)
        x = self.fc(x)
        x = x.view(-1, *self.output_shape)
        return x

class BenadjilaBest(nn.Module):
    def __init__(self, input_shape, output_shape, dropout=0.):
        super().__init__()
        
        def get_fe_block(in_channels, out_channels, kernel_size=11):
            return nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(num_features=out_channels),
                nn.ReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2))
        
        self.feature_extractor = nn.Sequential(
            get_fe_block(input_shape[0], 64),
            get_fe_block(64, 128),
            get_fe_block(128, 256),
            get_fe_block(256, 512),
            get_fe_block(512, 512))
        eg_input = torch.rand(1, *input_shape)
        eg_input = self.feature_extractor(eg_input)
        self.num_features = np.prod(eg_input.shape)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, 4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, np.prod(output_shape)))
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_features)
        x = self.mlp(x)
        x = x.view(-1, *self.output_shape)
        return x

class LeNet5(nn.Module):
    def __init__(self, input_shape, output_shape,
                 channels=[6, 16], hidden_neurons=[120, 84], dropout=0.):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=input_shape[0], out_channels=channels[0],
                      kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm1d(num_features=channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=channels[0], out_channels=channels[1],
                      kernel_size=5, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(num_features=channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        eg_input = torch.randn(1, *input_shape)
        eg_input = self.feature_extractor(eg_input)
        self.num_features = np.prod(eg_input.shape)
        self.mlp_probe = nn.Sequential(
            nn.Linear(self.num_features, hidden_neurons[0]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_neurons[0], hidden_neurons[1]),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_neurons[1], np.prod(output_shape)))
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def forward(self, x):
        x = x.view(-1, *self.input_shape)
        features = self.feature_extractor(x)
        features = features.view(-1, self.num_features)
        logits = self.mlp_probe(features)
        logits = logits.view(-1, *self.output_shape)
        return logits