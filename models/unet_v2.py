import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

class GlobalPool2d(nn.Module):
    def __init__(self, pool_fn=torch.mean):
        super().__init__()
        
        self.pool_fn = pool_fn
    
    def forward(self, x):
        out = self.pool_fn(x, dim=(2, 3))
        out = out.view(-1, out.size(1))
        return out
    
    def __repr__(self):
        return self.__class__.__name__+'({})'.format(self.pool_fn.__name__)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, groups=1,
                 bn=None, sn=spectral_norm, activation=lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        
        def db_conv2d(in_channels, out_channels, **kwargs):
            return sn(nn.Conv2d(in_channels, out_channels, groups=groups, **kwargs))
        
        self.residual_connection = []
        if bn is not None:
            self.residual_connection.append(bn(in_channels))
        self.residual_connection.append(activation())
        self.residual_connection.append(db_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if bn is not None:
            self.residual_connection.append(bn(out_channels))
        self.residual_connection.append(activation())
        self.residual_connection.append(db_conv2d(
            out_channels, out_channels, kernel_size=3, stride=2 if downsample else 1, padding=1
        ))
        self.residual_connection = nn.Sequential(*self.residual_connection)
        self.skip_connection = []
        if downsample:
            self.skip_connection.append(db_conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0))
        elif in_channels != out_channels:
            self.skip_connection.append(db_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        self.skip_connection = nn.Sequential(*self.skip_connection)
        
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.skip_connection(x)
        out = x_rc + x_sc
        return out
    
class Discriminator(nn.Module):
    def __init__(self, input_shape, activation=lambda: nn.LeakyReLU(0.1),
                 leakage_classes=2, initial_channels=8,
                 downsample_blocks=2, straight_blocks=2,
                 fe_groups=3, use_sn=True):
        super().__init__()
        
        sn = spectral_norm if use_sn else lambda x: x
        bn = None
        assert fe_groups in [
            1, # Realism and leakage classifiers share the same feature extractor
            2, # Realism and leakage classifiers have independent feature extractors
            3  # Realism and leakage classifiers have both independent and shared feature extractors
        ]
        self.fe_groups = fe_groups
        initial_channels_cnn = initial_channels if fe_groups==1 else 2*initial_channels if fe_groups==2 else 3*initial_channels//2
        
        self.input_transform = nn.Sequential(
            sn(nn.Conv2d(input_shape[0], initial_channels_cnn, kernel_size=1, stride=1, padding=0)),
            sn(nn.Conv2d(initial_channels_cnn, initial_channels_cnn, kernel_size=3, stride=1, padding=1, groups=fe_groups))
        )
        self.feature_extractor = []
        for n in range(downsample_blocks):
            self.feature_extractor.append(DiscriminatorBlock(
                initial_channels_cnn*2**n, initial_channels_cnn*2**(n+1),
                downsample=True, groups=fe_groups, sn=sn, bn=bn, activation=activation
            ))
        for _ in range(straight_blocks):
            self.feature_extractor.append(DiscriminatorBlock(
                initial_channels_cnn*2**downsample_blocks, initial_channels_cnn*2**downsample_blocks,
                downsample=False, groups=fe_groups, sn=sn, bn=bn, activation=activation
            ))
        self.feature_extractor.append(GlobalPool2d(pool_fn=torch.sum))
        self.feature_extractor = nn.Sequential(*self.feature_extractor)
        self.realism_classifier = sn(nn.Linear(2*initial_channels*2**downsample_blocks, 1))
        self.leakage_classifier = sn(nn.Linear(initial_channels*2**downsample_blocks, leakage_classes))
        
    def extract_features(self, x):
        x_i = self.input_transform(x)
        x_fe = self.feature_extractor(x_i)
        return x_fe
    
    def get_realism_features(self, x):
        if self.fe_groups == 1:
            return x
        elif self.fe_groups == 2:
            return x[:, :x.size(1)//2]
        elif self.fe_groups == 3:
            return x[:, :2*x.size(1)//3]
        else:
            assert False
    
    def get_leakage_features(self, x):
        if self.fe_groups == 1:
            return x
        elif self.fe_groups == 2:
            return x[:, x.size(1)//2:]
        elif self.fe_groups == 3:
            return x[:, x.size(1)//3:]
        else:
            assert False
    
    def classify_realism(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.realism_classifier(x)
    
    def classify_leakage(self, x):
        return self.leakage_classifier(x)
        
class Classifier(nn.Module):
    def __init__(self, input_shape, activation=lambda: nn.ReLU(inplace=True),
                 leakage_classes=2, initial_channels=8,
                 downsample_blocks=2, straight_blocks=2):
        super().__init__()
        
        sn = lambda x: x
        bn = nn.BatchNorm2d
        self.input_transform = nn.Conv2d(input_shape[0], initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.feature_extractor = []
        for n in range(downsample_blocks):
            self.feature_extractor.append(DiscriminatorBlock(
                initial_channels*2**n, initial_channels*2**(n+1),
                downsample=True, groups=1, sn=sn, bn=bn, activation=activation
            ))
        for _ in range(straight_blocks):
            self.feature_extractor.append(DiscriminatorBlock(
                initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks,
                downsample=False, groups=1, sn=sn, activation=activation
            ))
        self.feature_extractor.append(GlobalPool2d(pool_fn=torch.mean))
        self.feature_extractor = nn.Sequential(*self.feature_extractor)
        self.classifier = nn.Linear(initial_channels*2**downsample_blocks, leakage_classes)
        
    def forward(self, x):
        x_i = self.input_transform(x)
        x_fe = self.feature_extractor(x_i)
        out = self.classifier(x_fe)
        return out
        
class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 downsample=False, upsample=False, sn=spectral_norm, bn=nn.BatchNorm2d,
                 activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        
        def gb_conv2d(in_channels, out_channels, transpose=False, **kwargs):
            conv2d = nn.ConvTranspose2d if transpose else nn.Conv2d
            return sn(conv2d(in_channels, out_channels, **kwargs))
        
        self.residual_connection = []
        if bn is not None:
            self.residual_connection.append(bn(in_channels))
        self.residual_connection.append(activation())
        self.residual_connection.append(gb_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if bn is not None:
            self.residual_connection.append(bn(out_channels))
        self.residual_connection.append(activation())
        if upsample or downsample:
            self.residual_connection.append(gb_conv2d(
                out_channels, out_channels, kernel_size=2, stride=2, padding=0, transpose=upsample
            ))
        else:
            self.residual_connection.append(gb_conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1))
        self.residual_connection = nn.Sequential(*self.residual_connection)
        self.skip_connection = []
        if upsample or downsample:
            self.skip_connection.append(gb_conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, transpose=upsample))
        elif in_channels != out_channels:
            self.skip_connection.append(gb_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        self.skip_connection = nn.Sequential(*self.skip_connection)
        
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.skip_connection(x)
        out = x_rc + x_sc
        return out

class WrapWithResampler(nn.Module):
    def __init__(self, submodule, submodule_in_channels, submodule_out_channels, straight_blocks,
                 sn=spectral_norm, activation=lambda: nn.ReLU(inplace=True), bn=nn.BatchNorm2d):
        super().__init__()
        
        self.resample_path = nn.Sequential(
            GeneratorBlock(submodule_in_channels//2, submodule_in_channels,
                           downsample=True, sn=sn, bn=bn, activation=activation),
            submodule,
            GeneratorBlock(submodule_out_channels, submodule_in_channels//2,
                           upsample=True, sn=sn, bn=bn, activation=activation)
        )
        self.skip_path = nn.Sequential(*[
            GeneratorBlock(submodule_in_channels//2, submodule_in_channels//2,
                           sn=sn, bn=bn, activation=activation) for _ in range(straight_blocks)
           ])
        
    def forward(self, x):
        x_rp = self.resample_path(x)
        x_sp = self.skip_path(x)
        out = torch.cat((x_rp, x_sp), dim=1)
        return out

class Generator(nn.Module):
    def __init__(self, input_shape, initial_channels=8, resamples=2, 
                 straight_blocks_per_res=1, post_straight_blocks=1, use_sn=True, use_bn=True):
        super().__init__()
        
        sn = spectral_norm if use_sn else lambda x: x
        bn = nn.BatchNorm2d if use_bn else None
        activation = lambda: nn.ReLU(inplace=True)
        
        self.input_transform = sn(nn.Conv2d(input_shape[0], initial_channels, kernel_size=3, stride=1, padding=1))
        self.resampler = nn.Sequential(*[
            GeneratorBlock(initial_channels*2**resamples, initial_channels*2**resamples,
                           sn=sn, bn=bn, activation=activation) for _ in range(straight_blocks_per_res)
        ])
        for n in range(resamples):
            self.resampler = WrapWithResampler(
                self.resampler, initial_channels*2**(resamples-n), (1 if n==0 else 2)*initial_channels*2**(resamples-n),
                straight_blocks=straight_blocks_per_res, sn=sn, bn=bn, activation=activation
            )
        self.reconstructor = nn.Sequential(*[
            GeneratorBlock(2*initial_channels, initial_channels, sn=sn, bn=bn, activation=activation)
            for _ in range(post_straight_blocks-1)],
            GeneratorBlock((2 if post_straight_blocks==1 else 1)*initial_channels, input_shape[0], sn=sn, bn=bn, activation=activation)
        )
        
    def forward(self, x):
        x_i = self.input_transform(x)
        x_resampled = self.resampler(x_i)
        out = self.reconstructor(x_resampled)
        out = torch.tanh(out)
        return out