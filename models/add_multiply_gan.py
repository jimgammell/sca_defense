import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

class AddMultiplyBlock(nn.Module):
    def __init__(self, channels=32, activation=nn.ReLU, use_spectral_norm=True, input_block=False, output_block=False):
        super().__init__()
        self.input_block = input_block
        self.output_block = output_block
        
        def get_conv2d(*args, **kwargs):
            if use_spectral_norm:
                return spectral_norm(nn.Conv2d(*args, **kwargs))
            else:
                return nn.Conv2d(*args, **kwargs)
        
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                
                self.residual_block = nn.Sequential(
                    get_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    activation(),
                    get_conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                if in_channels != out_channels:
                    self.id = get_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                else:
                    self.id = nn.Identity()
                
            def forward(self, x):
                return self.id(x) + self.residual_block(x)
            
        if not input_block:
            self.xi_to_lambda_fn = ResidualBlock(channels, channels)
            self.hi_to_lambda_fn = ResidualBlock(channels, channels)
            self.xihi_to_lambda_fn = ResidualBlock(2*channels, channels)
        
        if not output_block:
            self.xi_to_ho_fn = ResidualBlock(channels, channels)
            self.hi_to_ho_fn = ResidualBlock(channels, channels)
            self.xihi_to_ho_fn = ResidualBlock(2*channels, channels)
        
    def forward(self, *args):
        def get_xo(xi, hi):
            xi_to_lambda = self.xi_to_lambda_fn(xi)
            hi_to_lambda = self.hi_to_lambda_fn(hi)
            xihi_to_lambda = torch.cat((xi_to_lambda, hi_to_lambda), dim=1)
            lmbda_logits = self.xihi_to_lambda_fn(xihi_to_lambda)
            lmbda = torch.sigmoid(lmbda_logits)
            xo = lmbda*xi + (1-lmbda)*hi
            return xo
        
        def get_ho(xi, hi):
            xi_to_ho = self.xi_to_ho_fn(xi)
            hi_to_ho = self.hi_to_ho_fn(hi)
            xihi_to_ho = torch.cat((xi_to_ho, hi_to_ho), dim=1)
            ho = self.xihi_to_ho_fn(xihi_to_ho)
            return ho
        
        if self.input_block:
            (xi,) = args
            xo = xi
            ho = get_ho(xi, xi)
            return xo, ho
        elif self.output_block:
            ((xi, hi),) = args
            xo = get_xo(xi, hi)
            return xo
        else:
            ((xi, hi),) = args
            xo = get_xo(xi, hi)
            ho = get_ho(xi, hi)
            return xo, ho
    
class Generator(nn.Module):
    def __init__(self, input_shape, channels=32, am_blocks=3, activation=nn.ReLU, use_spectral_norm=True):
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
        
        self.downsample_block = nn.Sequential(
            get_conv2d(input_shape[0], channels//2, kernel_size=2, stride=2, padding=0),
            get_conv2d(channels//2, channels, kernel_size=2, stride=2, padding=0)
        )
        
        self.am_blocks = nn.Sequential(
            AddMultiplyBlock(channels=channels, activation=activation, use_spectral_norm=use_spectral_norm, input_block=True),
          *[AddMultiplyBlock(channels=channels, activation=activation, use_spectral_norm=use_spectral_norm)
            for _ in range(am_blocks-2)],
            AddMultiplyBlock(channels=channels, activation=activation, use_spectral_norm=use_spectral_norm, output_block=True)
        )
        
        self.upsample_block = nn.Sequential(
            get_convtranspose2d(channels, channels//2, kernel_size=2, stride=2, padding=0),
            get_convtranspose2d(channels//2, input_shape[0], kernel_size=2, stride=2, padding=0),
            nn.Tanh()
        )
        
    def forward(self, x):
        x_ds = self.downsample_block(x)
        x_am = self.am_blocks(x_ds)
        x_us = self.upsample_block(x_am)
        return x_us

class Discriminator(nn.Module):
    def __init__(self, input_shape, channels=32, activation=nn.ReLU, use_spectral_norm=True):
        super().__init__()
        
        def get_conv2d(*args, **kwargs):
            if use_spectral_norm:
                return spectral_norm(nn.Conv2d(*args, **kwargs))
            else:
                return nn.Conv2d(*args, **kwargs)
        
        def get_linear(*args, **kwargs):
            if use_spectral_norm:
                return spectral_norm(nn.Linear(*args, **kwargs))
            else:
                return nn.Linear(*args, **kwargs)
        
        self.model = nn.Sequential(
            get_conv2d(input_shape[0], channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            get_conv2d(channels, 2*channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            get_conv2d(2*channels, 4*channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            get_conv2d(4*channels, 8*channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        eg_input = torch.randn(1, *input_shape)
        eg_output = self.model(eg_input)
        self.feature_distiller = get_conv2d(8*channels, 8*channels, kernel_size = eg_output.shape[2], stride=1, padding=0)
        self.realism_head = get_linear(8*channels, 1)
        self.matching_head = get_linear(16*channels, 1)
    
    def get_features(self, x):
        xf = self.model(x)
        features = self.feature_distiller(xf)
        features = features.view(-1, np.prod(features.shape[1:]))
        return features
        
    def assess_realism(self, features):
        realism = self.realism_head(features)
        return realism
    
    def assess_matching(self, features_A, features_B):
        features = torch.cat((features_A, features_B), dim=1)
        matching = self.matching_head(features)
        return matching