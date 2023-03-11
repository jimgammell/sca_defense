import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

class EndomorphicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_connection=False, activation=nn.ReLU):
        super().__init__()
        
        self.endomorphic_block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            activation(),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        )
        if skip_connection:
            if in_channels != out_channels:
                self.skip_connection = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
            else:
                self.skip_connection = nn.Identity()
        else:
            self.skip_connection = None
    
    def forward(self, x):
        output = self.endomorphic_block(x)
        if self.skip_connection is not None:
            output = output + self.skip_connection(x)
        return x

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

class Discriminator(nn.Module):
    def __init__(self, input_shape,
                 initial_channels=32,
                 downsample_blocks=2,
                 endomorphic_blocks=2,
                 activation=lambda: nn.LeakyReLU(0.2)):
        super().__init__()
        
        num_features = initial_channels*2**(downsample_blocks-1)
        self.feature_extractor = nn.Sequential(
            DownsampleBlock(2*input_shape[0], initial_channels, activation=activation),
          *[DownsampleBlock(initial_channels*2**n, initial_channels*2**(n+1), activation=activation)
            for n in range(downsample_blocks-1)],
          *[EndomorphicBlock(num_features, num_features, activation=activation, skip_connection=True)
            for _ in range(endomorphic_blocks)],
            spectral_norm(nn.Conv2d(num_features, num_features, kernel_size=input_shape[1]//(2**downsample_blocks)))
        )
        self.classifier = nn.Sequential(
            activation(),
            spectral_norm(nn.Linear(num_features, 1))
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(-1, np.prod(features.shape[1:]))
        output = self.classifier(features)
        return output
    
    def get_features(self, x):
        features = self.feature_extractor(x)
        features = features.view(-1, np.prod(features.shape[1:]))
        return features
    
    def assess_realism(self, features):
        assessment = self.classifier(features)
        return assessment
    
class Generator(nn.Module):
    def __init__(self, input_shape,
                 initial_channels=32,
                 resample_blocks=2,
                 endomorphic_blocks=2,
                 bottleneck_width=64,
                 activation=nn.ReLU):
        super().__init__()
        
        num_features = initial_channels*2**(resample_blocks-1)
        self.feature_extractor = nn.Sequential(
            DownsampleBlock(input_shape[0], initial_channels, activation=activation),
          *[DownsampleBlock(initial_channels*2**n, initial_channels*2**(n+1), activation=activation)
            for n in range(resample_blocks-1)],
          *[EndomorphicBlock(num_features, num_features, activation=activation, skip_connection=True)
            for _ in range(endomorphic_blocks)],
            spectral_norm(nn.Conv2d(num_features, num_features, kernel_size=input_shape[1]//(2**resample_blocks)))
        )
        self.bottleneck = nn.Sequential(
            spectral_norm(nn.Linear(num_features, bottleneck_width)),
            spectral_norm(nn.Linear(bottleneck_width, num_features))
        )
        self.feature_constructor = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(num_features, num_features, kernel_size=input_shape[1]//(2**resample_blocks))),
          *[EndomorphicBlock(num_features, num_features, activation=activation, skip_connection=True)
            for _ in range(endomorphic_blocks)],
          *[UpsampleBlock(num_features*2**n, num_features*2**(n+1), activation=activation)
            for n in range(resample_blocks-1)],
            UpsampleBlock(num_features*2**(resample_blocks-1), input_shape[0], activation=activation)
        )
        
    def forward(self, x):
        extracted_features = self.feature_extractor(x)
        fshape = extracted_features.shape[1:]
        extracted_features = extracted_features.view(-1, np.prod(extracted_features.shape[1:]))
        compressed_features = self.bottleneck(extracted_features)
        compressed_features = compressed_features.view(-1, *fshape)
        reconstructed_x = self.feature_constructor(compressed_features)
        reconstructed_x = torch.tanh(reconstructed_x)
        return reconstructed_x