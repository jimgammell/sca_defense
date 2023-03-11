# Based on this implementation: https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py

import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

def get_resample_layer(channels, fixed_resample=True, downsample=False, upsample=False):
    if downsample:
        assert not upsample
        if fixed_resample:
            resample = nn.AvgPool2d(2)
        else:
            resample = spectral_norm(nn.Conv2d(channels, channels, kernel_size=2, stride=2, padding=0))
    elif upsample:
        assert not downsample
        if fixed_resample:
            resample = nn.Upsample(scale_factor=2)
        else:
            resample = spectral_norm(nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, padding=0))
    else:
        resample = nn.Identity()
    return resample

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.query_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
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
    def __init__(self, in_channels, out_channels, fixed_resample=True, downsample=False, upsample=False):
        super().__init__()
        
        self.residual_connection = nn.Sequential(
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.1),
            spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            get_resample_layer(out_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample)
        )
        self.skip_connection = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)),
            get_resample_layer(out_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample)
        )
        self.rejoin = spectral_norm(nn.Conv2d(2*out_channels, out_channels, kernel_size=1, stride=1, padding=0))
        
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
    
class Discriminator(nn.Module):
    def __init__(self, input_shape, initial_channels=16, downsample_blocks=2, fixed_resample=True, two_inputs=True):
        super().__init__()
        if two_inputs:
            input_shape = [2*input_shape[0], *input_shape[1:]]
        
        self.input_transform = nn.Sequential(
            spectral_norm(nn.Conv2d(input_shape[0], initial_channels, kernel_size=1, stride=1, padding=0))
        )
        self.feature_extractor = nn.Sequential(
            DiscriminatorBlock(
                initial_channels, 2*initial_channels,
                fixed_resample=fixed_resample, downsample=True
            ),
            SelfAttentionBlock(2*initial_channels),
          *[DiscriminatorBlock(
              initial_channels*2**n, initial_channels*2**(n+1),
              fixed_resample=fixed_resample, downsample=True
            ) for n in range(1, downsample_blocks)],
            DiscriminatorBlock(
                initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks,
                fixed_resample=fixed_resample
            ),
            nn.LeakyReLU(0.1)
        )
        self.output_transform = nn.Sequential(
            spectral_norm(nn.Linear(initial_channels*2**downsample_blocks, 1))
        )
        
    def forward(self, x):
        x_i = self.input_transform(x)
        x_fe = self.feature_extractor(x_i)
        x_sp = x_fe.sum(dim=(2, 3)).view(-1, x_fe.shape[1])
        out = self.output_transform(x_sp)
        return out
        
class Generator(nn.Module):
    def __init__(self, input_shape, initial_channels=16, resample_blocks=2, fixed_resample=True):
        super().__init__()
        
        self.input_transform = nn.Sequential(
            spectral_norm(nn.Conv2d(input_shape[0], initial_channels, kernel_size=1, stride=1, padding=0))
        )
        self.feature_extractor = nn.Sequential(
            GeneratorBlock(
                initial_channels, 2*initial_channels,
                fixed_resample=fixed_resample, downsample=True),
            SelfAttentionBlock(2*initial_channels),
          *[GeneratorBlock(
              initial_channels*2**n, initial_channels*2**(n+1),
              fixed_resample=fixed_resample, downsample=True
            ) for n in range(1, resample_blocks)],
            GeneratorBlock(
                initial_channels*2**resample_blocks, initial_channels*2**resample_blocks,
                fixed_resample=fixed_resample
            )
        )
        self.reconstructor = nn.Sequential(
            GeneratorBlock(
                initial_channels*2**resample_blocks, initial_channels*2**resample_blocks,
                fixed_resample=fixed_resample, upsample=True
            ),
          *[GeneratorBlock(
              initial_channels*2**n, initial_channels*2**(n-1),
              fixed_resample=fixed_resample, upsample=True
            ) for n in range(resample_blocks, 1, -1)],
            SelfAttentionBlock(2*initial_channels),
            GeneratorBlock(
                2*initial_channels, initial_channels,
                fixed_resample=fixed_resample
            )
        )
        self.output_transform = nn.Sequential(
            GeneratorBlock(2*initial_channels, input_shape[0]),
            nn.Tanh()
        )
        
    def forward(self, x):
        x_i = self.input_transform(x)
        x_fe = self.feature_extractor(x_i)
        x_rec = self.reconstructor(x_fe)
        x_comb = torch.cat((x_i, x_rec), dim=1)
        out = self.output_transform(x_comb)
        return out