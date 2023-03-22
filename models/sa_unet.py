import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

def init_weights(gain):
    def _init_weights(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.orthogonal_(module.weight, gain=gain)
    return _init_weights

def get_resample_layer(channels, fixed_resample=True, downsample=False, upsample=False, sn=spectral_norm):
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

# Based on this implementation:
#   https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.query_conv = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = nn.functional.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)
        out = self.gamma*out + x
        return out

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fixed_resample=False, downsample=False, upsample=False, sn=spectral_norm, activation=lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        
        self.residual_connection = nn.Sequential(
            activation(),
            sn(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            activation(),
            sn(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            get_resample_layer(out_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample, sn=sn)
        )
        self.skip_connection = nn.Sequential(
            sn(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)),
            get_resample_layer(out_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample, sn=sn)
        )
        self.rejoin = sn(nn.Conv2d(2*out_channels, out_channels, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.skip_connection(x)
        x_comb = torch.cat((x_rc, x_sc), dim=1)
        out = self.rejoin(x_comb)
        return out

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fixed_resample=True, downsample=False, upsample=False):
        super().__init__()
        
        self.residual_connection = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            get_resample_layer(in_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        )
        self.skip_connection = nn.Sequential(
            get_resample_layer(in_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        )
        self.rejoin = spectral_norm(nn.Conv2d(2*out_channels, out_channels, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.skip_connection(x)
        x_comb = torch.cat((x_rc, x_sc), dim=1)
        out = self.rejoin(x_comb)
        return out
    
class SubmoduleWrapper(nn.Module):
    def __init__(self, submodule, submodule_channels, num_straight_blocks, sa_block=True, fixed_resample=True):
        super().__init__()
        
        self.resample_path = nn.Sequential(
            GeneratorBlock(submodule_channels//2, submodule_channels, downsample=True, fixed_resample=fixed_resample),
            submodule,
            GeneratorBlock(submodule_channels, submodule_channels//2, upsample=True, fixed_resample=fixed_resample)
        )
        self.skip_path = nn.Sequential(
          *[GeneratorBlock(submodule_channels//2, submodule_channels//2) for _ in range(num_straight_blocks//2)],
            SelfAttentionBlock(submodule_channels//2) if sa_block else nn.Identity(),
          *[GeneratorBlock(submodule_channels//2, submodule_channels//2) for _ in range(num_straight_blocks//2)]
        )
        self.rejoin = spectral_norm(nn.Conv2d(submodule_channels, submodule_channels//2, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x):
        x_rp = self.resample_path(x)
        x_sp = self.skip_path(x)
        x_comb = torch.cat((x_rp, x_sp), dim=1)
        out = self.rejoin(x_comb)
        return out
    
class Generator(nn.Module):
    def __init__(self, input_shape, initial_channels=8, downsample_blocks=2, sa_block=True, fixed_resample=True):
        super().__init__()
        
        self.input_transform = GeneratorBlock(input_shape[0], initial_channels)
        self.model = nn.Sequential(
            GeneratorBlock(initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks),
            SelfAttentionBlock(initial_channels*2**downsample_blocks) if sa_block else nn.Identity(),
            GeneratorBlock(initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks)
        )
        for n in range(downsample_blocks):
            self.model = SubmoduleWrapper(self.model, initial_channels*2**(downsample_blocks-n), 2,
                                          sa_block=sa_block, fixed_resample=fixed_resample)
        self.output_transform = GeneratorBlock(initial_channels, input_shape[0])
        self.apply(init_weights(nn.init.calculate_gain('relu')))
        
    def forward(self, x):
        x_i = self.input_transform(x)
        x_m = self.model(x_i)
        out = self.output_transform(x_m)
        out = torch.tanh(out)
        return out
    
class Discriminator(nn.Module):
    def __init__(self, input_shape, n_classes=1, initial_channels=8, downsample_blocks=2, sa_block=True, fixed_resample=True, use_sn=True, activation=lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        
        if use_sn:
            sn = spectral_norm
        else:
            sn = lambda x: x
        
        self.input_transform = DiscriminatorBlock(input_shape[0], initial_channels, sn=sn, activation=activation)
        self.feature_extractor = nn.Sequential(
            DiscriminatorBlock(initial_channels, 2*initial_channels,
                               fixed_resample=fixed_resample, downsample=True, sn=sn, activation=activation),
            SelfAttentionBlock(2*initial_channels) if sa_block else nn.Identity(),
            *[DiscriminatorBlock(initial_channels*2**n, initial_channels*2**(n+1),
                                 fixed_resample=fixed_resample, downsample=True, sn=sn, activation=activation)
              for n in range(1, downsample_blocks)],
            DiscriminatorBlock(initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks, sn=sn, activation=activation)
        )
        self.classifier = sn(nn.Linear(initial_channels*2**downsample_blocks, n_classes))
        self.apply(init_weights(nn.init.calculate_gain('leaky_relu', 0.1)))
    
    def extract_features(self, x):
        x_i = self.input_transform(x)
        x_fe = self.feature_extractor(x_i)
        x_sp = x_fe.sum(dim=(2, 3)).view(-1, x_fe.shape[1])
        return x_sp
    
    def classify_features(self, features):
        out = self.classifier(features)
        return out
    
    def forward(self, x):
        features = self.extract_features(x)
        out = self.classify_features(features)
        return out
    
class SanitizingDiscriminator(nn.Module):
    def __init__(self, input_shape, leakage_classes=2, fixed_resample=True):
        super().__init__()
        
        self.realism_analyzer = Discriminator([2*input_shape[0], *input_shape[1:]], n_classes=1, fixed_resample=fixed_resample)
        self.leakage_analyzer = Discriminator(input_shape, n_classes=leakage_classes, fixed_resample=fixed_resample)
        
    def assess_realism(self, eg_1, eg_2):
        return self.realism_analyzer(torch.cat((eg_1, eg_2), dim=1))
    
    def assess_leakage(self, eg):
        return self.leakage_analyzer(eg)