import numpy as np
import torch
from torch import nn

## Todo:
##   Try tanh instead of hardtanh
##     Reduced performance
##   Remove the multilayer perceptron blocks, try conv+global average pooling instead
##   Try removing/replacing batchnorm
##     Removing seems to lead to visually better samples, but doesn't always reproduce the watermarks
##   Try reducing network capacity to see if it makes things worse

class Encoder(nn.Module):
    class EncoderBlock(nn.Module):
        def __init__(self, input_shape, ds_factor, ce_factor):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(
                    input_shape[0],
                    ce_factor*input_shape[0],
                    kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.LayerNorm([ce_factor*input_shape[0], *list(input_shape[1:])]),
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    ce_factor*input_shape[0],
                    ce_factor*input_shape[0],
                    kernel_size=ds_factor, stride=ds_factor, padding=0, bias=False
                ),
                nn.LayerNorm([ce_factor*input_shape[0], *list(input_shape[1:]//ds_factor)])
            )
            
        def forward(self, x):
            return self.model(x)
    
    class ShortcutBlock(nn.Module):
        def __init__(self, input_shape, ds_factor, ce_factor):
            super().__init__()
            
            self.model = nn.Sequential(
                nn.Conv2d(input_shape[0], ce_factor*input_shape[0], kernel_size=ds_factor, stride=ds_factor, padding=0, bias=False),
                nn.LayerNorm([ce_factor*input_shape[0], *list(input_shape[1:]//ds_factor)])
            )
            
        def forward(self, x):
            return self.model(x)
    
    def __init__(self, 
                 input_shape=(1, 28, 28),
                 block0_channels=4,
                 num_blocks=2,
                 ds_factor=2,
                 ce_factor=4,
                 output_features=128,
                 use_sn=False):
        super().__init__()
        
        self.input_transform = nn.Conv2d(input_shape[0], block0_channels, 1, stride=1, padding=0)
        
        self.fe_modules = nn.ModuleDict({})
        shape = np.array(list(input_shape))
        shape[0] = block0_channels
        for block_idx in range(num_blocks):
            enc = Encoder.EncoderBlock(shape, ds_factor, ce_factor)
            sc = Encoder.ShortcutBlock(shape, ds_factor, ce_factor)
            shape[0] *= ce_factor
            shape[1:] //= ds_factor
            block = nn.ModuleDict({})
            block.update({'enc': enc})
            block.update({'sc': sc})
            self.fe_modules.update({'block_%d'%(block_idx): block})
        self.fe_modules.update({
            'compress': nn.Sequential(
                nn.Conv2d(shape[0], output_features, kernel_size=shape[1], stride=1, padding=0, bias=False)
            )
        })
        
        self.mixer = nn.Sequential(
            nn.LayerNorm([output_features, 1, 1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(output_features, output_features, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        xi = self.input_transform(x)
        for block_key, block_dict in self.fe_modules.items():
            if block_key == 'compress':
                continue
            enc, sc = block_dict.values()
            xi_enc = enc(xi)
            xi_sc = sc(xi)
            xi = xi_enc+xi_sc
        xi = self.fe_modules['compress'](xi)
        mixed_features = self.mixer(xi).view(-1, np.prod(xi.shape[1:]))
        return mixed_features

class Decoder(nn.Module):
    class DecoderBlock(nn.Module):
        def __init__(self, input_shape, us_factor, cr_factor):
            super().__init__()
            
            self.model = nn.Sequential(
                nn.LayerNorm(input_shape),
                nn.ConvTranspose2d(input_shape[0], input_shape[0], kernel_size=us_factor, stride=us_factor, padding=0, bias=False),
                nn.LayerNorm([input_shape[0], *list(input_shape[1:]*us_factor)]),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(input_shape[0], input_shape[0]//cr_factor, kernel_size=3, stride=1, padding=1, bias=False)
            )
            
        def forward(self, x):
            return self.model(x)
    
    class ShortcutBlock(nn.Module):
        def __init__(self, input_shape, us_factor, cr_factor):
            super().__init__()
            
            self.model = nn.Sequential(
                nn.LayerNorm(input_shape),
                nn.ConvTranspose2d(input_shape[0], input_shape[0]//cr_factor, kernel_size=us_factor, stride=us_factor, padding=0, bias=False)
            )
        
        def forward(self, x):
            return self.model(x)
        
    def __init__(self, input_shape=(1, 28, 28), num_blocks=2, us_factor=2, cr_factor=2, input_features=128, use_sn=False):
        super().__init__()
        
        shape = np.array(list(input_shape))
        shape[0] = input_features
        shape[1:] //= us_factor**num_blocks
        
        self.mixer = nn.Sequential(
            nn.LayerNorm([input_features, 1, 1]),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(input_features, input_features, kernel_size=1, stride=1, padding=0, bias=False)
        )
        
        self.fc_modules = nn.ModuleDict({})
        self.fc_modules.update({
            'expand': nn.Sequential(
                nn.LayerNorm([input_features, 1, 1]),
                nn.ConvTranspose2d(input_features, input_features//cr_factor, kernel_size=shape[1], stride=1, padding=0, bias=False)
            )
        })
        shape[0] //= cr_factor
        for block_idx in range(num_blocks):
            dec = Decoder.DecoderBlock(shape, us_factor, cr_factor)
            sc = Decoder.ShortcutBlock(shape, us_factor, cr_factor)
            shape[0] //= cr_factor
            shape[1:] *= us_factor
            block = nn.ModuleDict({})
            block.update({'dec': dec})
            block.update({'sc': sc})
            self.fc_modules.update({'block_%d'%(block_idx): block})
        
        self.output_transform = nn.Sequential(
            nn.LayerNorm(shape),
            nn.ConvTranspose2d(shape[0], input_shape[0], kernel_size=1, stride=1, padding=0),
            nn.Hardtanh()
        )
    
    def forward(self, x):
        xi = self.mixer(x.view(-1, np.prod(x.shape[1:]), 1, 1))
        xi = self.fc_modules['expand'](xi)
        for block_key, block_dict in self.fc_modules.items():
            if block_key == 'expand':
                continue
            enc, sc = block_dict.values()
            xi_enc = enc(xi)
            xi_sc = sc(xi)
            xi = xi_enc + xi_sc
        output = self.output_transform(xi)
        return output