import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

def init_orthogonal(model, gain):
    for m in model.modules():
        if any(isinstance(m, l) for l in (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=gain)

class Generator(nn.Module):
    def __init__(self, input_shape, initial_channels=64, residual_blocks=9, use_spectral_norm=True):
        super().__init__()
        
        def get_conv2d(*args, **kwargs):
            if use_spectral_norm:
                return spectral_norm(nn.Conv2d(*args, **kwargs))
            else:
                return nn.Conv2d(*args, **kwargs)
        
        def get_convtranspose2d(*args, **kwargs):
            if use_spectral_norm:
                return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))
            else:
                return nn.ConvTranspose2d(*args, **kwargs)
            
        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                
                self.residual_block = nn.Sequential(
                    get_conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    get_conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
                )
            
            def forward(self, x):
                return x + self.residual_block(x)
        
        self.downsample_block = nn.Sequential(
            get_conv2d(input_shape[0], initial_channels, kernel_size=5, stride=1, padding=3),
            nn.ReLU(),
            get_conv2d(initial_channels, 2*initial_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            get_conv2d(2*initial_channels, 4*initial_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(4*initial_channels) for _ in range(residual_blocks)
        ])
        
        self.upsample_block = nn.Sequential(
            get_convtranspose2d(4*initial_channels, 2*initial_channels, kernel_size=3, stride=2, padding=1),
            get_convtranspose2d(2*initial_channels, initial_channels, kernel_size=3, stride=2, padding=1),
            get_conv2d(initial_channels, input_shape[0], kernel_size=5, stride=1, padding=3)
        )
        
        gain = nn.init.calculate_gain('relu')
        init_orthogonal(self, gain)
        
    def forward(self, x):
        x_ds = self.downsample_block(x)
        x_p = self.residual_blocks(x_ds)
        x_r = self.upsample_block(x_p)
        return x_r

class Discriminator(nn.Module):
    def __init__(self, input_shape, initial_channels=64, use_spectral_norm=True):
        super().__init__()
        
        def get_conv2d(*args, **kwargs):
            if use_spectral_norm:
                return spectral_norm(nn.Conv2d(*args, **kwargs))
            else:
                return nn.Conv2d(*args, **kwargs)
        
        self.submodels = nn.Sequential(
            get_conv2d(input_shape[0], initial_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            get_conv2d(initial_channels, 2*initial_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            get_conv2d(2*initial_channels, 4*initial_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            get_conv2d(4*initial_channels, 8*initial_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            get_conv2d(8*initial_channels, 1, kernel_size=3, stride=1, padding=1)
        )
        
        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        init_orthogonal(self, gain)
    
    def forward(self, x):
        predictions = self.submodels(x)
        prediction = predictions.mean(dim=(1, 2, 3))
        return prediction