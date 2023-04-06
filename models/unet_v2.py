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

    
# Based on implementation here:
#   https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/sn_projection_cgan_64x64_143c.ipynb
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes, eps=2e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_classes, num_features))
            self.bias = nn.Parameter(torch.Tensor(num_classes, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()
    
    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
    
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.zero_()
    
    def forward(self, x, y):
        exponential_average_factor = 0.0
        
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:
                exponential_average_factor = self.momentum
        
        out = torch.nn.functional.batch_norm(
            x, self.running_mean, self.running_var, None, None,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)
        if self.affine:
            shape = [x.size(0), self.num_features] + (x.dim()-2)*[1]
            weight = self.weight.index_select(0, y).view(shape)
            bias = self.bias.index_select(0, y).view(shape)
            out = out*weight + bias
        return out
    
    def extra_repr(self):
        return '{num_features}, num_classes={num_classes}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, groups=1,
                 bn=None, sn=spectral_norm, activation=lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        
        def db_conv2d(in_channels, out_channels, **kwargs):
            return sn(nn.Conv2d(groups*in_channels, groups*out_channels, groups=groups, **kwargs))
        
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
    
class LeakageDiscriminator(nn.Module):
    def __init__(self, input_shape, leakage_classes,
                 activation=lambda: nn.LeakyReLU(0.1), initial_channels=8,
                 downsample_blocks=2, straight_blocks=2, use_sn=True):
        super().__init__()
        
        sn = spectral_norm if use_sn else lambda x: x
        bn = None
        
        self.input_transform = sn(nn.Conv2d(input_shape[0], 3*initial_channels, kernel_size=3, stride=1, padding=1))
        self.feature_extractor = []
        for n in range(downsample_blocks):
            self.feature_extractor.append(DiscriminatorBlock(
                initial_channels*2**n, initial_channels*2**(n+1),
                downsample=True, sn=sn, bn=bn, activation=activation, groups=3
            ))
        for n in range(straight_blocks):
            self.feature_extractor.append(DiscriminatorBlock(
                initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks,
                downsample=False, sn=sn, bn=bn, activation=activation, groups=3
            ))
        self.feature_extractor.append(GlobalPool2d(pool_fn=torch.sum))
        self.feature_extractor = nn.Sequential(*self.feature_extractor)
        self.leakage_classifier = sn(nn.Linear(2*initial_channels*2**downsample_blocks, leakage_classes))
        self.realism_classifier = sn(nn.Linear(2*initial_channels*2**downsample_blocks, 1))
    
    def extract_features(self, x):
        x_i = self.input_transform(x)
        return self.feature_extractor(x_i)
    
    def classify_leakage(self, x):
        return self.leakage_classifier(x[:, :2*x.size(1)//3])
    
    def classify_realism(self, x):
        return self.realism_classifier(x[:, x.size(1)//3:])
        
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
        
class ConditionalGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes,
                 downsample=False, upsample=False, sn=spectral_norm,
                 activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        
        def gb_conv2d(in_channels, out_channels, transpose=False, **kwargs):
            conv2d = nn.ConvTranspose2d if transpose else nn.Conv2d
            return sn(conv2d(in_channels, out_channels, **kwargs))
        
        self.rc_bn0 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.rc_act0 = activation()
        self.rc_cv0 = gb_conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.rc_bn1 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.rc_act1 = activation()
        if upsample or downsample:
            self.rc_cv1 = gb_conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=0, transpose=upsample)
        else:
            self.rc_cv1 = gb_conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if upsample or downsample:
            self.sc = gb_conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, transpose=upsample)
        elif in_channels != out_channels:
            self.sc = gb_conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.sc = nn.Sequential()
        
    def forward(self, x, y):
        x_rc = self.rc_bn0(x, y)
        x_rc = self.rc_act0(x_rc)
        x_rc = self.rc_cv0(x_rc)
        x_rc = self.rc_bn1(x_rc, y)
        x_rc = self.rc_act1(x_rc)
        x_rc = self.rc_cv1(x_rc)
        x_sc = self.sc(x)
        out = x_rc + x_sc
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
    def __init__(self, submodule, submodule_in_channels, submodule_out_channels, num_classes, straight_blocks,
                 sn=spectral_norm, activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        
        self.rp_i = ConditionalGeneratorBlock(submodule_in_channels//2, submodule_in_channels, num_classes,
                                              downsample=True, sn=sn, activation=activation)
        self.rp_sm = submodule
        self.rp_o = ConditionalGeneratorBlock(submodule_out_channels, submodule_in_channels//2, num_classes,
                                              upsample=True, sn=sn, activation=activation)
        self.sp = nn.ModuleList([
            ConditionalGeneratorBlock(submodule_in_channels//2, submodule_in_channels//2, num_classes,
                           sn=sn, activation=activation) for _ in range(straight_blocks)
        ])
        
    def forward(self, x, y):
        x_rp = self.rp_i(x, y)
        x_rp = self.rp_sm(x_rp, y)
        x_rp = self.rp_o(x_rp, y)
        x_sp = x
        for sp_mod in self.sp:
            x_sp = sp_mod(x_sp, y)
        out = torch.cat((x_rp, x_sp), dim=1)
        return out

class Generator(nn.Module):
    def __init__(self, input_shape, num_leakage_classes=1, initial_channels=16, resamples=2, 
                 straight_blocks_per_res=1, post_straight_blocks=1, use_sn=True, use_bn=True):
        super().__init__()
        self.num_leakage_classes = num_leakage_classes
        
        sn = spectral_norm if use_sn else lambda x: x
        activation = lambda: nn.ReLU(inplace=True)
        
        if self.num_leakage_classes > 1:
            self.context_transform = sn(nn.ConvTranspose2d(self.num_leakage_classes, 1,
                                                           kernel_size=input_shape[1], stride=1, padding=0))
        self.input_transform = sn(nn.Conv2d(input_shape[0] + (1 if self.num_leakage_classes>1 else 0),
                                            initial_channels, kernel_size=3, stride=1, padding=1))
        self.resampler = ConditionalGeneratorBlock(initial_channels*2**resamples, initial_channels*2**resamples, num_leakage_classes,
                                                   sn=sn, activation=activation)
        for n in range(resamples):
            self.resampler = WrapWithResampler(
                self.resampler, initial_channels*2**(resamples-n), (1 if n==0 else 2)*initial_channels*2**(resamples-n),
                num_leakage_classes, straight_blocks=straight_blocks_per_res, sn=sn, activation=activation
            )
        self.reconstructor = nn.ModuleList([
            ConditionalGeneratorBlock(2*initial_channels, initial_channels, num_leakage_classes, sn=sn, activation=activation)
            for _ in range(post_straight_blocks-1)] + [
            ConditionalGeneratorBlock((2 if post_straight_blocks==1 else 1)*initial_channels, input_shape[0], num_leakage_classes, sn=sn, activation=activation)
        ])
        
    def forward(self, x, y):
        if self.num_leakage_classes > 1:
            context = torch.tensor([[1.0 if j==y_b else 0.0 for j in range(self.num_leakage_classes)]
                                    for y_b in y], dtype=torch.float, device=y.device).view(y.size(0), self.num_leakage_classes, 1, 1)
            context = self.context_transform(context)
            x_i = self.input_transform(torch.cat((x, context), dim=1))
        else:
            x_i = self.input_transform(x)
        x_resampled = self.resampler(x_i, y)
        for rec_mod in self.reconstructor:
            x_resampled = rec_mod(x_resampled, y)
        out = torch.tanh(x_resampled)
        return out