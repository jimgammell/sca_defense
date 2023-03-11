import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        
        self.downsample_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            activation(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0))
        )
        
    def forward(self, x):
        return self.downsample_block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        
        self.upsample_block = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)),
            activation(),
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        )
        
    def forward(self, x):
        return self.upsample_block(x)

class StraightBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        
        self.straight_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            activation(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        )
        if in_channels != out_channels:
            self.skip_connection = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        else:
            self.skip_connection = nn.Identity()
        
    def forward(self, x):
        x_sb = self.straight_block(x)
        x_sc = self.skip_connection(x)
        output = x_sc + x_sb
        return output

class WrapSubmodule(nn.Module):
    def __init__(self, submodule, submodule_channels, num_straight_blocks, activation=nn.ReLU):
        super().__init__()
        
        self.resample_path = nn.Sequential(
            DownsampleBlock(submodule_channels//2, submodule_channels, activation=activation),
            submodule,
            UpsampleBlock(submodule_channels, submodule_channels//2, activation=activation)
        )
        self.skip_path = nn.Sequential(*[
            StraightBlock(submodule_channels//2, submodule_channels//2, activation=activation)
            for _ in range(num_straight_blocks)
        ])
        
    def forward(self, x):
        x_rp = self.resample_path(x)
        x_sp = self.skip_path(x)
        output = x_rp + x_sp
        return output

class Generator(nn.Module):
    def __init__(self, input_shape, initial_channels=32, downsample_blocks=2):
        super().__init__()
        
        self.input_transform = spectral_norm(nn.Conv2d(input_shape[0], initial_channels, kernel_size=1, stride=1, padding=0))
        self.model = nn.Sequential(
            StraightBlock(initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks),
            StraightBlock(initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks)
        )
        for n in range(downsample_blocks):
            self.model = WrapSubmodule(self.model, initial_channels*2**(downsample_blocks-n), 2*n+4)
        self.output_transform = StraightBlock(initial_channels, input_shape[0])
        
    def forward(self, x):
        xi = self.input_transform(x)
        xm = self.model(xi)
        output = self.output_transform(xm)
        output = torch.tanh(output)
        return output

class Discriminator(nn.Module):
    def __init__(self, input_shape, initial_channels=32, downsample_blocks=2, straight_blocks=2):
        super().__init__()
        
        num_features = initial_channels*2**downsample_blocks
        self.input_transform = spectral_norm(nn.Conv2d(2*input_shape[0], initial_channels, kernel_size=1, stride=1, padding=0))
        self.feature_extractor = nn.Sequential(
          *[DownsampleBlock(initial_channels*2**n, initial_channels*2**(n+1), activation=lambda: nn.LeakyReLU(0.2))
            for n in range(downsample_blocks)],
          *[StraightBlock(num_features, num_features, activation=lambda: nn.LeakyReLU(0.2))
            for _ in range(straight_blocks)],
            spectral_norm(nn.Conv2d(num_features, num_features, kernel_size=input_shape[1]//(2**downsample_blocks)))
        )
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(num_features, 1))
        )
        
    def forward(self, x):
        xi = self.input_transform(x)
        x_fe = self.feature_extractor(xi)
        x_fe = x_fe.view(-1, np.prod(x_fe.shape[1:]))
        output = self.classifier(x_fe)
        return output