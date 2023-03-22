import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

def init_weights(gain):
    def _init_weights(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.orthogonal_(module.weight, gain=gain)
    return _init_weights

def get_resample_layer(channels, fixed_resample=True, downsample=False, upsample=False, use_sn=True):
    if use_sn:
        sn = spectral_norm
    else:
        sn = lambda x: x
    if downsample:
        assert not upsample
        if fixed_resample:
            resample = nn.AvgPool2d(2)
        else:
            resample = sn(nn.Conv2d(channels, channels, kernel_size=2, stride=2, padding=0))
    elif upsample:
        assert not downsample
        if fixed_resample:
            resample = nn.Upsample(scale_factor=2)
        else:
            resample = sn(nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, padding=0))
    else:
        resample = nn.Identity()
    return resample

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fixed_resample=False, downsample=False, upsample=False, use_sn=True, activation=lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        if use_sn:
            sn = spectral_norm
        else:
            sn = lambda x: x
        
        self.residual_connection = nn.Sequential(
            sn(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)),
            activation(),
            get_resample_layer(out_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample, use_sn=use_sn)
        )
        
    def forward(self, x):
        return self.residual_connection(x)

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fixed_resample=True, downsample=False, upsample=False):
        super().__init__()
        
        self.residual_connection = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            get_resample_layer(in_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2))
        )
        
    def forward(self, x):
        return self.residual_connection(x)
    
class Discriminator(nn.Module):
    def __init__(self, input_shape,
                 binary=False, n_classes=2, fixed_resample=False, use_sn=True, activation=lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        self.binary = binary
        
        if use_sn:
            sn = spectral_norm
        else:
            sn = lambda x: x
        
        self.feature_extractor = nn.Sequential(
            DiscriminatorBlock(input_shape[0], 16, fixed_resample=fixed_resample, downsample=True, use_sn=use_sn, activation=activation),
            DiscriminatorBlock(16, 32, fixed_resample=fixed_resample, downsample=True, use_sn=use_sn, activation=activation),
            nn.Flatten(),
            sn(nn.Linear(32*(input_shape[1]//4)**2, 120)),
            activation(),
            sn(nn.Linear(120, 84)),
            activation()
        )
        self.feature_classifier = nn.Sequential(
            sn(nn.Linear(2*84 if binary else 84, n_classes))
        )
        
    def extract_features(self, x):
        return self.feature_extractor(x)
    
    def classify_features(self, features):
        return self.feature_classifier(features)
            
    def forward(self, *args):
        if self.binary:
            (x1, x2) = args
            features = torch.cat(
                (self.extract_features(x1), self.extract_features(x2)), dim=1
            )
        else:
            (x,) = args
            features = self.extract_features(x)
        out = self.classify_features(features)
        return out
    
class SanitizingDiscriminator(nn.Module):
    def __init__(self, input_shape, leakage_classes=2, bottleneck_width=64, fixed_resample=False):
        super().__init__()
        
        self.realism_analyzer = Discriminator(
            input_shape, binary=True, n_classes=1, fixed_resample=fixed_resample
        )
        self.leakage_analyzer = Discriminator(
            input_shape, binary=False, n_classes=leakage_classes, fixed_resample=fixed_resample
        )
        #self.apply(init_weights(nn.init.calculate_gain('leaky_relu', 0.1)))
        
    def assess_realism(self, eg_1, eg_2):
        return self.realism_analyzer(torch.cat((eg_1, eg_2), dim=1))
    
    def assess_leakage(self, eg):
        out = self.leakage_analyzer(eg)
        return out
    
class Generator(nn.Module):
    def __init__(self, input_shape, bottleneck_width=64, fixed_resample=False, skip_connection=False):
        super().__init__()
        
        self.input_transform = GeneratorBlock(input_shape[0], 16)
        self.feature_extractor = nn.Sequential(
            GeneratorBlock(16, 32, fixed_resample=fixed_resample, downsample=True),
            GeneratorBlock(32, 64, fixed_resample=fixed_resample, downsample=True),
            nn.ReLU(),
            #nn.Flatten(),
            #spectral_norm(nn.Linear(64*(input_shape[1]//4)**2, bottleneck_width))
        )
        self.feature_reconstructor = nn.Sequential(
            #nn.ReLU(),
            #spectral_norm(nn.Linear(bottleneck_width, 64*(input_shape[1]//4)**2)),
            #nn.Unflatten(1, (64, input_shape[1]//4, input_shape[1]//4)),
            GeneratorBlock(64, 32, fixed_resample=fixed_resample, upsample=True),
            GeneratorBlock(32, 16, fixed_resample=fixed_resample, upsample=True)
        )
        if skip_connection:
            self.skip_connection = GeneratorBlock(input_shape[0], 16)
        else:
            self.skip_connection = None
        self.output_transformation = nn.Sequential(
            GeneratorBlock(32 if skip_connection else 16, input_shape[0]),
            nn.Tanh()
        )
        #self.apply(init_weights(nn.init.calculate_gain('relu')))
        
    def forward(self, x):
        x_i = self.input_transform(x)
        x_fe = self.feature_extractor(x_i)
        x_fr = self.feature_reconstructor(x_fe)
        if self.skip_connection is not None:
            x_sc = self.skip_connection(x)
            x_o = torch.cat((x_fr, x_sc), dim=1)
        else:
            x_o = x_fr
        out = self.output_transformation(x_o)
        return out