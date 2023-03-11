import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

class Generator(nn.Module):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(np.prod(input_shape), 64)),
            nn.ReLU(),
            spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            spectral_norm(nn.Linear(64, 64)),
            nn.ReLU(),
            spectral_norm(nn.Linear(64, np.prod(input_shape)))
        )
        
    def forward(self, x):
        x = x.view(-1, np.prod(x.shape[1:]))
        x = self.model(x)
        x = x.view(-1, *self.input_shape)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_shape, *args, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        
        self.feature_extractor = nn.Sequential(
            spectral_norm(nn.Linear(np.prod(input_shape), 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 64)),
            nn.LeakyReLU(0.2)
        )
        self.classifier = spectral_norm(nn.Linear(64, 1))
        
    def forward(self, x):
        x = x.view(-1, np.prod(x.shape[1:]))
        features = self.feature_extractor(x)
        prediction = self.classifier(features)
        return prediction