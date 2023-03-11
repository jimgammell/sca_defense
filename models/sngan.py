import numpy as np
import torch
from torch import nn

## Todo: feed 2 examples (real + reconstructed) to discriminator and get it to say which is the real one

def get_conv2d(*args, use_spectral_norm=False, **kwargs):
    if use_spectral_norm:
        return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
    else:
        return nn.Conv2d(*args, **kwargs)

def get_convtranspose2d(*args, use_spectral_norm=False, **kwargs):
    if use_spectral_norm:
        return nn.utils.spectral_norm(nn.ConvTranspose2d(*args, **kwargs))
    else:
        return nn.ConvTranspose2d(*args, **kwargs)

def init_orthogonal(model):
    for m in model.modules():
        if any(isinstance(m, l) for l in (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal(m.weight)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=nn.ReLU(),
                 downsample=True,
                 output_activation=True,
                 use_instance_norm=False,
                 use_spectral_norm=False):
        super().__init__()
    
        modules = []
        modules.append(get_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if use_instance_norm:
            modules.append(nn.InstanceNorm2d(out_channels))
        if output_activation:
            modules.append(activation())
        if downsample:
            modules.append(get_conv2d(out_channels, out_channels, kernel_size=4, stride=2))
            if use_instance_norm:
                modules.append(nn.InstanceNorm2d(out_channels))
        self.model = nn.Sequential(*modules)
        
    def forward(self, x):
        return self.model(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation=nn.ReLU,
                 use_instance_norm=False,
                 use_spectral_norm=False):
        super().__init__()

        modules = []
        modules.append(get_convtranspose2d(in_channels, out_channels, kernel_size=4, stride=2))
        if use_instance_norm:
            modules.append(nn.InstanceNorm2d(out_channels))
        modules.append(activation())
        self.model = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.model(x)

class FeatureExtractor(nn.Module):
    def __init__(self, input_shape,
                 downsample_blocks=3,
                 num_kernels=64,
                 activation=nn.ReLU,
                 use_instance_norm=False,
                 use_spectral_norm=False):
        super().__init__()
        
        input_channels = input_shape[0]
        kwargs = {'activation': activation, 'use_instance_norm': use_instance_norm, 'use_spectral_norm': use_spectral_norm}
        self.feature_extractor = nn.Sequential(
            DownsampleBlock(input_channels, num_kernels, **kwargs),
          *[DownsampleBlock(num_kernels*2**i, num_kernels*2**(i+1), **kwargs) for i in range(downsample_blocks-1)],
            get_conv2d(num_kernels*2**(downsample_blocks-1), num_kernels*2**downsample_blocks, kernel_size=3, padding=1, use_spectral_norm=use_spectral_norm)
        )
        eg_input = torch.randn(1, *input_shape)
        eg_output = self.feature_extractor(eg_input)
        self.uncompressed_shape = eg_output.shape[1:]
        self.feature_compressor = nn.Conv2d(num_kernels*2**downsample_blocks, num_kernels*2**downsample_blocks, kernel_size=eg_output.shape[2], stride=1, padding=0)
        if use_spectral_norm:
            self.feature_compressor = nn.utils.spectral_norm(self.feature_compressor)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        compressed_features = self.feature_compressor(features)
        compressed_features = compressed_features.view(-1, np.prod(compressed_features.shape[1:]))
        return compressed_features

class FeatureConstructor(nn.Module):
    def __init__(self, input_shape, output_channels, upsample_blocks=3, num_kernels=64, activation=nn.ReLU, use_instance_norm=False, use_spectral_norm=False):
        super().__init__()
        
        self.feature_decompressor = nn.ConvTranspose2d(input_shape[0], input_shape[0], input_shape[1], stride=1, padding=0)
        if use_spectral_norm:
            self.feature_decompressor = nn.utils.spectral_norm(self.feature_decompressor)
        kwargs = {'activation': activation, 'use_instance_norm': use_instance_norm, 'use_spectral_norm': use_spectral_norm}
        self.feature_constructor = nn.Sequential(
            UpsampleBlock(input_shape[0], num_kernels*2**(upsample_blocks-1), **kwargs),
          *[UpsampleBlock(num_kernels*2**(upsample_blocks-i-1), num_kernels*2**(upsample_blocks-i-2), **kwargs)
            for i in range(upsample_blocks-1)],
            get_convtranspose2d(num_kernels, output_channels, kernel_size=3, use_spectral_norm=use_spectral_norm),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = x.view(-1, x.shape[1], 1, 1)
        decompressed_features = self.feature_decompressor(x)
        reconstructed_input = self.feature_constructor(decompressed_features)
        return reconstructed_input

class Discriminator(nn.Module):
    def __init__(self, input_shape, downsample_blocks=2, num_kernels=64, use_instance_norm=False, use_spectral_norm=False):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_shape,
            downsample_blocks=downsample_blocks,
            num_kernels=num_kernels,
            activation=lambda: nn.LeakyReLU(0.2),
            use_instance_norm=use_instance_norm,
            use_spectral_norm=use_spectral_norm
        )
        eg_input = torch.randn(1, *input_shape)
        eg_output = self.feature_extractor(eg_input)
            
        num_features = eg_output.shape[1]
        self.classifier = nn.Linear(2*num_features, 1)
        init_orthogonal(self)
        
    def get_features(self, x):
        features = self.feature_extractor(x)
        return features
    
    def forward(self, x1, x2):
        f1 = self.get_features(x1)
        f2 = self.get_features(x2)
        f = torch.cat((f1, f2), dim=-1)
        prediction = self.classifier(f)
        return prediction
    
class Generator(nn.Module):
    def __init__(self, input_shape, upsample_blocks=3, num_kernels=64, bottleneck_width=128, use_instance_norm=False, use_spectral_norm=False):
        super().__init__()
        
        self.feature_extractor = FeatureExtractor(
            input_shape,
            downsample_blocks=upsample_blocks,
            num_kernels=num_kernels,
            activation=nn.ReLU,
            use_instance_norm=use_instance_norm,
            use_spectral_norm=use_spectral_norm
        )
        uncompressed_shape = self.feature_extractor.uncompressed_shape
        num_features = uncompressed_shape[0]
        if use_spectral_norm:
            self.bottleneck = nn.utils.spectral_norm(nn.Linear(num_features, bottleneck_width))
            self.invbottleneck = nn.utils.spectral_norm(nn.Linear(bottleneck_width, num_features))
        else:
            self.bottleneck = nn.Linear(num_features, bottleneck_width)
            self.invbottleneck = nn.Linear(bottleneck_width, num_features)
        self.feature_constructor = FeatureConstructor(
            uncompressed_shape, input_shape[0],
            upsample_blocks=upsample_blocks,
            num_kernels=num_kernels,
            use_instance_norm=use_instance_norm,
            use_spectral_norm=use_spectral_norm
        )
        init_orthogonal(self)
        
    def get_features(self, x):
        features = self.feature_extractor(x)
        compressed_features = self.bottleneck(features)
        return compressed_features
    
    def reconstruct_features(self, features):
        decompressed_features = self.invbottleneck(features)
        reconstructed_input = self.feature_constructor(decompressed_features)
        return reconstructed_input
    
    def forward(self, x):
        features = self.get_features(x)
        reconstruction = self.reconstruct_features(features)
        return reconstruction