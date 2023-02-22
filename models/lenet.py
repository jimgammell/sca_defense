import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

class LeNet5Classifier(nn.Module):
    def __init__(self, input_channels=1, output_classes=10, use_sn=False):
        super().__init__()
        
        if use_sn:
            sn = spectral_norm
        else:
            sn = lambda x: x
        
        feature_extractor = nn.Sequential(
            sn(nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=0)),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            sn(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        eg_input = torch.randn(1, input_channels, 28, 28)
        eg_output = feature_extractor(eg_input)
        classifier = nn.Sequential(
            sn(nn.Linear(np.prod(eg_output.shape), 120)),
            nn.ReLU(),
            sn(nn.Linear(120, 84)),
            nn.ReLU(),
            sn(nn.Linear(84, output_classes)))
        self.model = nn.Sequential(
            feature_extractor,
            nn.Flatten(),
            classifier)
        
    def forward(self, x):
        return self.model(x)

class LeNet5Autoencoder(nn.Module):
    def __init__(self, shape=(1, 28, 28), use_sn=False, mixer_width=128):
        super().__init__()
        self.shape = shape
        if use_sn:
            sn = spectral_norm
        else:
            sn = lambda x: x
        
        self.feature_extractor = nn.Sequential(
            sn(nn.Conv2d(shape[0], 6, kernel_size=5, stride=1, padding=0)),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            sn(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.mixer = nn.Sequential(
            sn(nn.Linear(256, mixer_width)),
            nn.ReLU(),
            sn(nn.Linear(mixer_width, 256)))
        self.reconstructor = nn.Sequential(
            nn.Upsample(scale_factor=2),
            sn(nn.ConvTranspose2d(16, 6, kernel_size=5, stride=1, padding=0)),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            sn(nn.ConvTranspose2d(6, 6, kernel_size=5, stride=1, padding=0)),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            sn(nn.ConvTranspose2d(6, 1, kernel_size=1, stride=1, padding=0)))
    
    def forward(self, x):
        features = self.feature_extractor(x).view(-1, 256)
        mixed_features = self.mixer(features)
        reconstructed_input = self.reconstructor(mixed_features.view(-1, 16, 4, 4))
        return reconstructed_input