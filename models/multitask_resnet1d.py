import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

def apply_sn(modules):
    for idx, module in modules:
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            modules[idx] = spectral_norm(module)

class GlobalPool1d(nn.Module):
    def __init__(self, pool_fn=torch.mean):
        super().__init__()
        
        self.pool_fn = pool_fn
    
    def forward(self, x):
        out = self.pool_fn(x, dim=-1)
        out = out.view(-1, out.size(1))
        return out
    
    def __repr__(self):
        return self.__class__.__name__+'({})'.format(self.pool_fn.__name__)

class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features, class_embedding_size, eps=1e-4, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.num_features = num_features
        self.class_embedding_size = class_embedding_size
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.class_to_weight_transform = spectral_norm(nn.Linear(class_embedding_size, num_features), eps=1e-4)
            self.class_to_bias_transform = spectral_norm(nn.Linear(class_embedding_size, num_features), eps=1e-4)
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
            weight = self.class_to_weight_transform(y).view(shape)
            bias = self.class_to_bias_transform(y).view(shape)
            out = out*weight + bias
        return out
    
    def extra_repr(self):
        return '{num_features}, class_embedding_size={class_embedding_size}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}'.format(**self.__dict__)

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, groups=1,
                 bn=None, sn=spectral_norm, activation=lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        
        def db_conv1d(in_channels, out_channels, **kwargs):
            return sn(nn.Conv1d(groups*in_channels, groups*out_channels, groups=groups, **kwargs))
        
        residual_modules = []
        if bn is not None:
            residual_modules.append(bn(in_channels))
        residual_modules.append(activation())
        residual_modules.append(db_conv1d(in_channels, out_channels//4, kernel_size=1, stride=1, padding=0))
        if bn is not None:
            residual_modules.append(bn(out_channels//4))
        residual_modules.append(activation())
        residual_modules.append(db_conv1d(out_channels//4, out_channels//4, kernel_size=3, stride=1 if not downsample else 2, padding=1))
        if bn is not None:
            residual_modules.append(bn(out_channels//4))
        residual_modules.append(activation())
        residual_modules.append(db_conv1d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0))
        self.residual_connection = nn.Sequential(*residual_modules)
        
        skip_modules = []
        if downsample:
            skip_modules.append(db_conv1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0))
        elif in_channels != out_channels:
            skip_modules.append(db_conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        self.skip_connection = nn.Sequential(*skip_modules)
        
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.skip_connection(x)
        if x_sc.size(-1) > x_rc.size(-1):
            x_sc = x_sc[:, :, :x_rc.size(-1)]
        elif x_sc.size(-1) < x_rc.size(-1):
            pad_size = x_rc.size(-1)-x_sc.size(-1)
            x_sc = nn.functional.pad(x_sc, (pad_size//2, pad_size-pad_size//2), mode='constant', value=x_sc.mean().detach().item())
        out = x_rc + x_sc
        return out

class LeakageDiscriminator(nn.Module):
    def __init__(self, input_shape, head_sizes, activation=lambda: nn.LeakyReLU(0.1),
                 initial_channels=16, downsample_blocks=4, straight_blocks_per_downsample_block=3, use_sn=True):
        super().__init__()
        
        sn = lambda x: spectral_norm(x, eps=1e-4) if use_sn else lambda x: x
        bn = None
        
        channels_per_group = initial_channels//2
        total_channels = 3*channels_per_group
        self.class_embedding = sn(nn.Linear(sum(head_sizes.values()), 1024))
        self.input_transform = sn(nn.Conv1d(input_shape[0], total_channels, kernel_size=3, stride=1, padding=1))
        feature_extractor_modules = []
        for n in range(downsample_blocks):
            feature_extractor_modules.append(DiscriminatorBlock(
                total_channels*2**n, total_channels*2**(n+1),
                downsample=True, sn=sn, bn=bn, activation=activation
            ))
            for _ in range(straight_blocks_per_downsample_block):
                feature_extractor_modules.append(DiscriminatorBlock(
                    total_channels*2**(n+1), total_channels*2**(n+1),
                    downsample=False, sn=sn, bn=bn, activation=activation
                ))
        feature_extractor_modules.append(GlobalPool1d(pool_fn=torch.sum))
        self.feature_extractor = nn.Sequential(*feature_extractor_modules)
        
        shared_head_modules = []
        shared_head_modules.append(sn(nn.Linear(total_channels*2**downsample_blocks, 1024)))
        shared_head_modules.append(activation())
        self.shared_head = nn.Sequential(*shared_head_modules)
        
        self.classifier_heads = nn.ModuleDict({})
        for head_key, head_size in head_sizes.items():
            head = sn(nn.Linear(1024, head_size))
            self.classifier_heads.update({str(head_key): head})
            
        self.realism_head = sn(nn.Linear(1024, 1))
        
    def extract_features(self, x):
        x_i = self.input_transform(x)
        out = self.feature_extractor(x_i)
        return out
    
    def classify_leakage(self, x):
        x_sh = self.shared_head(x)
        out = {head_key: head(x_sh) for head_key, head in self.classifier_heads.items()}
        return out
    
    def classify_realism(self, x, y):
        x = self.shared_head(x)
        y = torch.cat(list(y.values()), dim=-1).to(torch.float)
        embedded_y = self.class_embedding(y)
        out = self.realism_head(x) + (x*embedded_y).sum(dim=-1, keepdim=True)
        return out
    
class ConditionalGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, class_embedding_size,
                 downsample=False, upsample=False, sn=spectral_norm,
                 activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        
        def gb_conv1d(in_channels, out_channels, transpose=False, **kwargs):
            conv1d = nn.ConvTranspose1d if transpose else nn.Conv1d
            return sn(conv1d(in_channels, out_channels, **kwargs))
        
        self.rc_bn0 = ConditionalBatchNorm(in_channels, class_embedding_size)
        self.rc_act0 = activation()
        self.rc_cv0 = gb_conv1d(in_channels, out_channels//4, kernel_size=1, stride=1, padding=0)
        self.rc_bn1 = ConditionalBatchNorm(out_channels//4, class_embedding_size)
        self.rc_act1 = activation()
        if upsample or downsample:
            self.rc_cv1 = gb_conv1d(out_channels//4, out_channels//4, kernel_size=3, stride=2, padding=1, transpose=upsample)
        else:
            self.rc_cv1 = gb_conv1d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1)
        self.rc_bn2 = ConditionalBatchNorm(out_channels//4, class_embedding_size)
        self.rc_act2 = activation()
        self.rc_cv2 = gb_conv1d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0)
        if upsample or downsample:
            self.sc = gb_conv1d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, transpose=upsample)
        elif in_channels != out_channels:
            self.sc = gb_conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.sc = nn.Sequential()
    
    def forward(self, x, y):
        x_rc = self.rc_bn0(x, y)
        x_rc = self.rc_act0(x_rc)
        x_rc = self.rc_cv0(x_rc)
        x_rc = self.rc_bn1(x_rc, y)
        x_rc = self.rc_act1(x_rc)
        x_rc = self.rc_cv1(x_rc)
        x_rc = self.rc_bn2(x_rc, y)
        x_rc = self.rc_act2(x_rc)
        x_rc = self.rc_cv2(x_rc)
        x_sc = self.sc(x)
        if x_sc.size(-1) > x_rc.size(-1):
            x_sc = x_sc[:, :, :x_rc.size(-1)]
        elif x_sc.size(-1) < x_rc.size(-1):
            pad_size = x_rc.size(-1)-x_sc.size(-1)
            x_sc = nn.functional.pad(x_sc, (pad_size//2, pad_size-pad_size//2), mode='constant', value=x_sc.mean().detach().item())
        out = x_rc + x_sc
        return out

class WrapWithResampler(nn.Module):
    def __init__(self, submodule, submodule_in_channels, submodule_out_channels, class_embedding_size,
                 straight_blocks, sn=spectral_norm, activation=lambda: nn.ReLU(inplace=True)):
        super().__init__()
        
        self.rp_i = ConditionalGeneratorBlock(submodule_in_channels//2, submodule_in_channels, class_embedding_size,
                                              downsample=True, sn=sn, activation=activation)
        self.rp_sm = submodule
        self.rp_o = ConditionalGeneratorBlock(submodule_out_channels, submodule_in_channels//2, class_embedding_size,
                                              upsample=True, sn=sn, activation=activation)
        self.sp = nn.ModuleList([
            ConditionalGeneratorBlock(submodule_in_channels//2, submodule_in_channels//2, class_embedding_size,
                                      sn=sn, activation=activation) for _ in range(straight_blocks)
        ])
        
    def forward(self, x, y):
        x_rp = self.rp_i(x, y)
        x_rp = self.rp_sm(x_rp, y)
        x_rp = self.rp_o(x_rp, y)
        if x_rp.size(-1) > x.size(-1):
            x_rp = x_rp[:, :, :x.size(-1)]
        elif x_rp.size(-1) < x.size(-1):
            pad_size = x.size(-1)-x_rp.size(-1)
            x_rp = nn.functional.pad(x_rp, (pad_size//2, pad_size-pad_size//2), mode='constant', value=x_rp.mean().detach().item())
        x_sp = x
        for sp_mod in self.sp:
            x_sp = sp_mod(x_sp, y)
        out = torch.cat((x_rp, x_sp), dim=1)
        return out

class Generator(nn.Module):
    def __init__(self, input_shape, head_sizes, initial_channels=32, resamples=4,
                 straight_blocks_per_resample=3, post_straight_blocks=3, use_sn=True, use_bn=True):
        super().__init__()
        
        class Seq2i(nn.Sequential):
            def forward(self, x, y):
                for module in self:
                    x = module(x, y)
                return x
        
        sn = lambda x: spectral_norm(x, eps=1e-4) if use_sn else lambda x: x
        activation = lambda: nn.ReLU(inplace=True)
        class_embedding_size = 1024
        self.class_embedding_size = class_embedding_size
        
        self.class_embedding = sn(nn.Linear(sum(head_sizes.values()), class_embedding_size))
        self.class_conditional_bias = sn(nn.ConvTranspose1d(
            class_embedding_size, input_shape[0], kernel_size=input_shape[1], stride=1, padding=0
        ))
        self.input_transform = sn(nn.Conv1d(
            input_shape[0], initial_channels, kernel_size=3, stride=1, padding=1
        ))
        self.resampler = Seq2i(*[
            ConditionalGeneratorBlock(
                initial_channels*2**resamples, initial_channels*2**resamples,
                class_embedding_size, sn=sn, activation=activation
            ) for _ in range(straight_blocks_per_resample)
        ])
        for n in range(resamples):
            self.resampler = WrapWithResampler(
                self.resampler, initial_channels*2**(resamples-n), (1 if n==0 else 2)*initial_channels*2**(resamples-n),
                class_embedding_size, straight_blocks=straight_blocks_per_resample, sn=sn, activation=activation
            )
        self.reconstructor = nn.ModuleList([
            ConditionalGeneratorBlock(2*initial_channels, initial_channels, class_embedding_size, sn=sn, activation=activation),
          *[ConditionalGeneratorBlock(initial_channels, initial_channels, class_embedding_size, sn=sn, activation=activation)
            for _ in range(post_straight_blocks-1)]
        ])
        self.output_transform = sn(nn.Conv1d(initial_channels, input_shape[0], kernel_size=1, stride=1, padding=0))
    
    def forward(self, x, y):
        y = torch.cat(list(y.values()), dim=-1).to(torch.float)
        embedded_y = self.class_embedding(y)
        class_conditional_bias = self.class_conditional_bias(embedded_y.view(x.size(0), self.class_embedding_size, 1))
        x_i = self.input_transform(x + class_conditional_bias)
        x_resampled = self.resampler(x_i, embedded_y)
        for rec_mod in self.reconstructor:
            x_resampled = rec_mod(x_resampled, embedded_y)
        out = torch.tanh(self.output_transform(x_resampled))
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1,
                 activation=lambda: nn.ReLU(inplace=True),
                 use_bn=False, use_sn=False):
        super().__init__()
        
        residual_modules = []
        if use_bn:
            residual_modules.append(nn.BatchNorm1d(in_channels))
        residual_modules.append(activation())
        residual_modules.append(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1, stride=1, padding=0, bias=False)
        )
        if use_bn:
            residual_modules.append(nn.BatchNorm1d(out_channels//4))
        residual_modules.append(activation())
        residual_modules.append(
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        )
        if use_bn:
            residual_modules.append(nn.BatchNorm1d(out_channels//4))
        residual_modules.append(activation())
        residual_modules.append(
            nn.Conv1d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=0)
        )
        if use_sn:
            apply_sn(residual_modules)
        
        shortcut_modules = []
        if stride != 1:
            shortcut_modules.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=kernel_size//2)
            )
        elif in_channels != out_channels:
            shortcut_modules.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            )
        if use_sn:
            apply_sn(shortcut_modules)
        
        self.residual_connection = nn.Sequential(*residual_modules)
        self.shortcut_connection = nn.Sequential(*shortcut_modules)
    
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.shortcut_connection(x)
        if x_sc.size(-1) > x_rc.size(-1):
            x_sc = x_sc[:, :, :x_rc.size(-1)]
        elif x_sc.size(-1) < x_rc.size(-1):
            x_sc = nn.functional.pad(x_sc, x_rc.size(-1)-x_sc.size(-1), mode='constant', value=0)
        out = x_rc + x_sc
        return out

class Classifier(nn.Module):
    def __init__(self, input_shape, head_sizes,
                 initial_filters=32, block_kernel_size=3,
                 activation=lambda: nn.ReLU(inplace=True), dense_dropout=0.1,
                 num_blocks=[3, 4, 4, 3], use_bn=True, use_sn=False):
        super().__init__()
        
        def get_stack(in_channels, out_channels, blocks,
                      kernel_size=3, stride=2, activation=lambda: nn.ReLU(inplace=True),
                      use_bn=False, use_sn=False):
            modules = []
            modules.append(
                ResidualBlock(in_channels, out_channels, kernel_size=kernel_size,
                              activation=activation, use_sn=use_sn, use_bn=use_bn)
            )
            for _ in range(2, blocks):
                modules.append(
                    ResidualBlock(out_channels, out_channels, kernel_size=kernel_size,
                                  activation=activation, use_sn=use_sn, use_bn=use_bn)
                )
            modules.append(
                ResidualBlock(out_channels, out_channels, kernel_size=kernel_size,
                              activation=activation, stride=stride, use_sn=use_sn, use_bn=use_bn)
            )
            stack = nn.Sequential(*modules)
            return stack
        
        self.input_transform = nn.Conv1d(input_shape[0], initial_filters, kernel_size=3, stride=1, padding=1, bias=False)
        if use_sn:
            self.input_transform = spectral_norm(input_transform)
        
        feature_extractor_modules = []
        filters = initial_filters
        for block_idx in range(4):
            filters *= 2
            feature_extractor_modules.append(
                get_stack(filters//2, filters, num_blocks[block_idx],
                          kernel_size=block_kernel_size, activation=activation,
                          use_bn=use_bn, use_sn=use_sn)
            )
        self.feature_extractor = nn.Sequential(*feature_extractor_modules)
        
        shared_head_modules = []
        shared_head_modules.append(nn.Dropout(dense_dropout))
        shared_head_modules.append(nn.Linear(filters, 256, bias=False))
        if use_bn:
            shared_head_modules.append(nn.BatchNorm1d(256))
        shared_head_modules.append(activation())
        if use_sn:
            apply_sn(shared_head_modules)
        self.shared_head = nn.Sequential(*shared_head_modules)
        
        self.heads = nn.ModuleDict({})
        for head_key, head_size in head_sizes.items():
            if use_sn:
                head = spectral_norm(nn.Linear(256, head_size))
            else:
                head = nn.Linear(256, head_size)
            self.heads.update({str(head_key): head})
    
    def forward(self, x):
        x_i = self.input_transform(x)
        x_fe = self.feature_extractor(x_i)
        features = torch.mean(x_fe, dim=2).view(x_fe.size(0), x_fe.size(1))
        features_sh = self.shared_head(features)
        logits = {head_name: head(features_sh) for head_name, head in self.heads.items()}
        return logits