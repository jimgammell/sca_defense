import numpy as np
import torch
from torch import nn

class EffNet_N100(nn.Module):
    def __init__(self, input_shape, simplified=False):
        super().__init__()
        
        self.input_shape = input_shape
        self.simplified = simplified
        
        fe_mods = []
        if not simplified:
            fe_mods.extend([
                nn.Conv1d(1, 32, 1),
                nn.SELU(),
                nn.BatchNorm1d(32)])
        fe_mods.extend([
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, 50, padding=50//2),
            nn.SELU(),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(50),
            nn.Conv1d(64, 128, 3, padding=3//2),
            nn.SELU(),
            nn.BatchNorm1d(128),
            nn.AvgPool1d(2)])
        self.feature_extractor = nn.Sequential(*fe_mods)
        eg_input = torch.randn(1, *input_shape)
        self.num_features = np.prod(eg_input.shape[1:])
        self.fc = nn.Sequential(
            nn.Linear(self.num_features, 20),
            nn.SELU(),
            nn.Linear(20, 20),
            nn.SELU(),
            nn.Linear(20, 20),
            nn.SELU(),
            nn.Linear(20, 256))
        
    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        logits = self.fc(features)
        return logits

class EffCnn_N50(nn.Module):
    def __init__(self, input_shape, simplified=False):
        super().__init__()
        
        self.input_shape = input_shape
        self.simplified = simplified
        
        fe_mods = []
        if not simplified:
            fe_mods.extend([
                nn.Conv1d(1, 32, 1),
                nn.SELU(),
                nn.BatchNorm1d(32)])
        fe_mods.extend([
            nn.AvgPool1d(2),
            nn.Conv1d(32, 64, 25, padding=25//2),
            nn.SELU(),
            nn.BatchNorm1d(64),
            nn.AvgPool1d(25),
            nn.Conv1d(64, 128, 3, padding=3//2),
            nn.SELU(),
            nn.BatchNorm1d(128),
            nn.AvgPool1d(4)])
        self.feature_extractor = nn.Sequential(*fe_mods)
        eg_input = torch.randn(1, *input_shape)
        self.num_features = np.prod(eg_input.shape[1:])
        self.fc = nn.Sequential(
            nn.Linear(self.num_features, 15),
            nn.SELU(),
            nn.Linear(15, 15),
            nn.SELU(),
            nn.Linear(15, 15),
            nn.SELU(),
            nn.Linear(16, 256))
    
    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        logits = self.fc(features)
        return logits

class EffCnn_N0(nn.Module):
    def __init__(self, input_shape, simplified=False):
        super().__init__()
        
        self.input_shape = input_shape
        self.simplified = simplified
        
        fe_mods = []
        if simplified:
            fe_mods.extend([
                nn.Conv1d(1, 4, 1),
                nn.SELU(),
                nn.BatchNorm1d(4)])
        fe_mods.extend([
            nn.AvgPool1d(2)])
        self.feature_extractor = nn.Sequential(*fe_mods)
        eg_input = torch.randn(1, *input_shape)
        self.num_features = np.prod(eg_input.shape[1:])
        self.fc = nn.Sequential(
            nn.Linear(self.num_features, 10),
            nn.SELU(),
            nn.Linear(10, 10),
            nn.SELU(),
            nn.Linear(10, 256))
        
    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        logits = self.fc(features)
        return logits