from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

# Based on this implementation:
#   https://github.com/clovaai/stargan-v2/blob/master/core/model.py
class AdaptiveInstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, embedding_dim, num_features, *args, use_sn=False, **kwargs):
        assert embedding_dim is not None
        super().__init__(num_features, *args, affine=False, **kwargs)
        self.get_affine = nn.Linear(embedding_dim, 2*num_features)
        self.use_sn = use_sn
    
    def forward(self, x, y):
        batch_size, channels, _ = x.size()
        affine_params = self.get_affine(y)
        gamma, beta = torch.split(affine_params, channels, dim=1)
        gamma, beta = gamma.view(batch_size, channels, 1), beta.view(batch_size, channels, 1)
        x_norm = super().forward(x)
        if self.use_sn:
            scalar = torch.clip(1 + gamma, -1, 1)
        else:
            scalar = 1 + gamma
        out = scalar * x_norm + beta
        return out
    
class SnBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, use_sn=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sn = use_sn
        
    def forward(self, *args, **kwargs):
        if self.weight is not None and self.use_sn:
            self.weight.data = torch.clip(self.weight.data, -1, 1)
        return super().forward(*args, **kwargs)

class SnInstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, *args, use_sn=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sn = use_sn
        
    def forward(self, *args, **kwargs):
        if self.weight is not None and self.use_sn:
            self.weight.data = torch.clip(self.weight.data, -1, 1)
        return super().forward(*args, **kwargs)
    
class ResidBlockBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_modules = nn.ModuleList([])
        self.shortcut_modules = nn.ModuleList([])
    
    def forward(self, *args):
        if len(args) == 1:
            (x,) = args
            y = None
        elif len(args) == 2:
            (x, y) = args
        else:
            raise NotImplementedError
            
        x_resid = x
        for resid_mod in self.residual_modules:
            if isinstance(resid_mod, AdaptiveInstanceNorm1d):
                assert y is not None
                x_resid = resid_mod(x_resid, y)
            else:
                x_resid = resid_mod(x_resid)
                
        x_sc = x
        for sc_mod in self.shortcut_modules:
            x_sc = sc_mod(x_sc)
        
        assert x_resid.size(0) == x_sc.size(0) and x_resid.size(1) == x_sc.size(1)
        if x_resid.size(2) < x_sc.size(2):
            x_sc = x_sc[:, :, :x_resid.size(2)]
        elif x_resid.size(2) > x_sc.size(2):
            pad_size = x_resid.size(2) - x_sc.size(2)
            x_sc = nn.functional.pad(x_sc, (pad_size//2, (pad_size//2)+(1 if pad_size%2!=0 else 0)))
        
        out = (x_resid + x_sc) / np.sqrt(2)
        return out
    
class ResNetBlock(ResidBlockBase):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 embedding_dim=None,
                 activation=None,
                 use_sn=False,
                 resample_layer='none',
                 norm_layer=None):
        
        super().__init__()
        
        if norm_layer is not None:
            self.residual_modules.append(norm_layer(in_channels))
        if activation is not None:
            self.residual_modules.append(activation())
        self.residual_modules.append(nn.Conv1d(in_channels, in_channels//4, kernel_size=1))
        if norm_layer is not None:
            self.residual_modules.append(norm_layer(in_channels//4))
        if activation is not None:
            self.residual_modules.append(activation())
        if resample_layer == 'none':
            self.residual_modules.append(nn.Conv1d(
                in_channels//4, in_channels//4, kernel_size=kernel_size, padding=kernel_size//2
            ))
        elif resample_layer == 'downsample':
            self.residual_modules.append(nn.Conv1d(
                in_channels//4, in_channels//4, kernel_size=kernel_size, stride=2, padding=kernel_size//2
            ))
        elif resample_layer == 'upsample':
            self.residual_modules.append(nn.ConvTranspose1d(
                in_channels//4, in_channels//4, kernel_size=kernel_size, stride=2, padding=kernel_size//2
            ))
        else:
            raise NotImplementedError
        if norm_layer is not None:
            self.residual_modules.append(norm_layer(in_channels//4))
        if activation is not None:
            self.residual_modules.append(activation())
        self.residual_modules.append(nn.Conv1d(in_channels//4, out_channels, kernel_size=1))
        
        if resample_layer is not None or in_channels != out_channels:
            if resample_layer == 'none':
                self.shortcut_modules.append(nn.Conv1d(
                    in_channels, out_channels, kernel_size=1
                ))
            elif resample_layer == 'downsample':
                self.shortcut_modules.append(nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=2
                ))
            elif resample_layer == 'upsample':
                self.shortcut_modules.append(nn.ConvTranspose1d(
                    in_channels, out_channels, kernel_size=1, stride=2
                ))
        
        if use_sn:
            for mod_idx, mod in enumerate(self.residual_modules):
                if isinstance(mod, nn.Conv1d):
                    self.residual_modules[mod_idx] = spectral_norm(mod)
            for mod_idx, mod in enumerate(self.shortcut_modules):
                if isinstance(mod, nn.Conv1d):
                    self.shortcut_modules[mod_idx] = spectral_norm(mod)
    
class ConvNextBlock(ResidBlockBase):
    def __init__(self, in_channels, out_channels,
                 kernel_size=7,
                 embedding_dim=None,
                 activation=None,
                 use_sn=False,
                 resample_layer='none',
                 norm_layer=None):
        
        super().__init__()
        
        if resample_layer == 'none':
            self.residual_modules.append(nn.Conv1d(
                in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, padding=kernel_size//2
            ))
        elif resample_layer == 'downsample':
            self.residual_modules.append(nn.Conv1d(
                in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2
            ))
        elif resample_layer == 'upsample':
            self.residual_modules.append(nn.ConvTranspose1d(
                in_channels, in_channels, groups=in_channels, kernel_size=kernel_size, stride=2, padding=kernel_size//2
            ))
        else:
            raise NotImplementedError
        if norm_layer is not None:
            self.residual_modules.append(norm_layer(in_channels))
        self.residual_modules.append(nn.Conv1d(in_channels, 4*in_channels, kernel_size=1))
        if activation is not None:
            self.residual_modules.append(activation())
        self.residual_modules.append(nn.Conv1d(4*in_channels, out_channels, kernel_size=1))
        
        if resample_layer is not None or in_channels != out_channels:
            if resample_layer == 'none':
                self.shortcut_modules.append(nn.Conv1d(
                    in_channels, out_channels, kernel_size=1
                ))
            elif resample_layer == 'downsample':
                self.shortcut_modules.append(nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=2
                ))
            elif resample_layer == 'upsample':
                self.shortcut_modules.append(nn.ConvTranspose1d(
                    in_channels, out_channels, kernel_size=1, stride=2
                ))
        
        if use_sn:
            for mod_idx, mod in enumerate(self.residual_modules):
                if isinstance(mod, nn.Conv1d):
                    self.residual_modules[mod_idx] = spectral_norm(mod)
            for mod_idx, mod in enumerate(self.shortcut_modules):
                if isinstance(mod, nn.Conv1d):
                    self.shortcut_modules[mod_idx] = spectral_norm(mod)
    
class StarGanResidualBlock(ResidBlockBase):
    def __init__(self, in_channels, out_channels,
                 kernel_size=7,
                 embedding_dim=None,
                 use_sn=False,
                 activation=None,
                 resample_layer='none',
                 norm_layer=None):
        
        super().__init__()
        
        if norm_layer is not None:
            self.residual_modules.append(norm_layer(in_channels))
        if activation is not None:
            self.residual_modules.append(activation())
        self.residual_modules.append(nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size//2))
        if resample_layer == 'downsample':
            self.residual_modules.append(nn.AvgPool1d(2))
        elif resample_layer == 'upsample':
            self.residual_modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif resample_layer == 'none':
            pass
        else:
            raise NotImplementedError
        if norm_layer is not None:
            self.residual_modules.append(norm_layer(in_channels))
        if activation is not None:
            self.residual_modules.append(activation())
        self.residual_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
        
        if in_channels != out_channels:
            self.shortcut_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        if resample_layer == 'downsample':
            self.shortcut_modules.append(nn.AvgPool1d(2))
        elif resample_layer == 'upsample':
            self.shortcut_modules.append(nn.Upsample(scale_factor=2, mode='nearest'))
        
        if use_sn:
            for mod_idx, mod in enumerate(self.residual_modules):
                if isinstance(mod, nn.Conv1d):
                    self.residual_modules[mod_idx] = spectral_norm(mod)
            for mod_idx, mod in enumerate(self.shortcut_modules):
                if isinstance(mod, nn.Conv1d):
                    self.shortcut_modules[mod_idx] = spectral_norm(mod)
    
class ModelBase(nn.Module):
    def __init__(self, activation='relu', norm_layer=None, block='convnext', use_sn=False, embedding_dim=None,
                 us_norm_layer=None, ds_norm_layer=None):
        super().__init__()
        
        self.activation_constructor = {
            'relu': lambda: nn.ReLU(inplace=True),
            'leaky_relu': lambda: nn.LeakyReLU(0.1),
            'gelu': lambda: nn.GELU()
        }[activation]
        norm_layer_constructor_dict = {
            'none':                   None,
            'batch_norm':             lambda features: SnBatchNorm1d(features, affine=True, use_sn=use_sn),
            'instance_norm':          lambda features: SnInstanceNorm1d(features, affine=True, use_sn=use_sn),
            'adaptive_instance_norm': lambda features: AdaptiveInstanceNorm1d(embedding_dim, features, use_sn=use_sn)
        }
        if norm_layer is not None:
            self.norm_layer_constructor = norm_layer_constructor_dict[norm_layer]
        if us_norm_layer is not None:
            self.us_norm_layer_constructor = norm_layer_constructor_dict[us_norm_layer]
        if ds_norm_layer is not None:
            self.ds_norm_layer_constructor = norm_layer_constructor_dict[ds_norm_layer]
        self.block_constructor = {
            'convnext': ConvNextBlock,
            'resnet': ResNetBlock,
            'stargan': StarGanResidualBlock
        }[block]
    
class Generator(ModelBase):
    def __init__(self, input_shape, head_sizes, base_channels=16, max_channels=128,
                 ds_blocks=4, iso_blocks=4,
                 kernel_size=3, activation='relu', embedding_dim=128,
                 ds_norm_layer='instance_norm', us_norm_layer='adaptive_instance_norm',
                 block='stargan', use_sn=True):
        super().__init__(activation=activation, embedding_dim=embedding_dim,
                         ds_norm_layer=ds_norm_layer, us_norm_layer=us_norm_layer,
                         use_sn=use_sn, block=block)
        
        self.get_ch = lambda n: min((base_channels*2**n, max_channels))
        apply_sn_fn = spectral_norm if use_sn else lambda x: x
        self.head_sizes = head_sizes
        self.input_transform = apply_sn_fn(nn.Conv1d(in_channels=input_shape[0], out_channels=self.get_ch(0), kernel_size=1, stride=1, padding=0))
        self.downsample_blocks = nn.ModuleList([
            self.block_constructor(
                self.get_ch(n), self.get_ch(n+1),
                kernel_size=kernel_size, activation=self.activation_constructor,
                resample_layer='downsample', norm_layer=self.ds_norm_layer_constructor, use_sn=use_sn
            ) for n in range(ds_blocks)
        ])
        self.iso_blocks = nn.ModuleList([
            self.block_constructor(
                self.get_ch(ds_blocks), self.get_ch(ds_blocks),
                kernel_size=kernel_size, activation=self.activation_constructor,
                resample_layer='none', norm_layer=self.ds_norm_layer_constructor, use_sn=use_sn
            ) for n in range(iso_blocks//2)] + [
            self.block_constructor(
                self.get_ch(ds_blocks), self.get_ch(ds_blocks),
                kernel_size=kernel_size, activation=self.activation_constructor,
                resample_layer='none', norm_layer=self.us_norm_layer_constructor, use_sn=use_sn
            ) for n in range((iso_blocks//2)+(1 if iso_blocks%2!=0 else 0))
        ])
        self.upsample_blocks = nn.ModuleList([
            self.block_constructor(
                self.get_ch(n), self.get_ch(n-1),
                kernel_size=kernel_size, activation=self.activation_constructor,
                resample_layer='upsample', norm_layer=self.us_norm_layer_constructor, use_sn=use_sn
            ) for n in range(ds_blocks, 0, -1)
        ])
        self.output_transform = apply_sn_fn(nn.Conv1d(in_channels=self.get_ch(0), out_channels=input_shape[0], kernel_size=1, stride=1, padding=0))
        self.class_embedding = apply_sn_fn(nn.Linear(sum(head_sizes.values()), embedding_dim))
        
    def embed(self, x):
        x = deepcopy(x)
        for (x_name, x_val), head_size in zip(x.items(), self.head_sizes.values()):
            if len(x_val.shape) < 2:
                x[x_name] = nn.functional.one_hot(x_val, head_size)
        x = torch.cat(list(x.values()), dim=-1).to(torch.float)
        embedded_x = self.class_embedding(x)
        return embedded_x
        
    def forward(self, x, y):
        embedded_y = self.embed(y)
        x = self.input_transform(x)
        x_lengths = []
        for ds_block in self.downsample_blocks:
            x = ds_block(x)
            x_lengths.append(x.size(-1))
        for iso_block in self.iso_blocks:
            x = iso_block(x, embedded_y)
        for us_block, l in zip(self.upsample_blocks, x_lengths[::-1]):
            if x.size(2) < l:
                pad_amt = l - x.size(2)
                x = nn.functional.pad(x, (pad_amt//2, (pad_amt//2)+(1 if pad_amt%2!=0 else 0)))
            x = us_block(x, embedded_y)
        x = self.output_transform(x)
        x = torch.tanh(x)
        return x
    
class UnetGenerator(Generator):
    def __init__(self, *args, use_sn=True, base_channels=16, **kwargs):
        super().__init__(*args, use_sn=use_sn, base_channels=base_channels, **kwargs)
        self.recombine_layers = nn.ModuleList([
            nn.Conv1d(2*self.get_ch(n), self.get_ch(n), kernel_size=1)
            for n in range(len(self.downsample_blocks)+1)
        ][::-1])
        if use_sn:
            for mod_idx, mod in enumerate(self.recombine_layers):
                self.recombine_layers[mod_idx] = spectral_norm(mod)
    
    def forward(self, x, y):
        embedded_y = self.embed(y)
        x = self.input_transform(x)
        x_lengths = []
        x_scales = [x.clone()]
        for ds_block in self.downsample_blocks:
            x = ds_block(x)
            x_lengths.append(x.size(-1))
            x_scales.append(x.clone())
        x_scales, x_lengths = x_scales[::-1], x_lengths[::-1]
        for iso_block in self.iso_blocks:
            x = iso_block(x, embedded_y)
        x = torch.cat((x, x_scales[0]), dim=1)
        x = self.recombine_layers[0](x)
        for us_block, x_scale, x_length, recomb_mod in zip(self.upsample_blocks, x_scales[1:], x_lengths, self.recombine_layers[1:]):
            x = us_block(x, embedded_y)
            if x.size(2) < x_scale.size(2):
                pad_amt = x_scale.size(2) - x.size(2)
                x = nn.functional.pad(x, (pad_amt//2, (pad_amt//2)+(1 if pad_amt%2!=0 else 0)))
            x = torch.cat((x, x_scale), dim=1)
            x = recomb_mod(x)
        x = self.output_transform(x)
        x = torch.tanh(x)
        return x
        
class Classifier(ModelBase):
    def __init__(self, input_shape, head_sizes, base_channels=16,
                 downsample_blocks=4, iso_blocks_per_res=3,
                 kernel_size=7, activation='relu',
                 norm_layer='instance_norm', block='convnext'):
        super().__init__(activation=activation, norm_layer=norm_layer, block=block, use_sn=False)
        
        self.head_sizes = head_sizes
        self.input_transform = nn.Conv1d(input_shape[0], base_channels, kernel_size=1)
        get_ds_block = lambda in_channels, out_channels: self.block_constructor(
            in_channels, out_channels, activation=self.activation_constructor, resample_layer='downsample',
            norm_layer=self.norm_layer_constructor, use_sn=False, kernel_size=kernel_size)
        get_iso_block = lambda in_channels, out_channels: self.block_constructor(
            in_channels, out_channels, activation=self.activation_constructor, resample_layer='none',
            norm_layer=self.norm_layer_constructor, use_sn=False, kernel_size=kernel_size)
        
        feature_extractor_modules = []
        for _ in range(downsample_blocks):
            base_channels *= 2
            feature_extractor_modules.append(get_ds_block(base_channels//2, base_channels))
            for _ in range(iso_blocks_per_res):
                feature_extractor_modules.append(get_iso_block(base_channels, base_channels))
        eg_input = torch.randn(1, *input_shape)
        eg_input = self.input_transform(eg_input)
        for mod in feature_extractor_modules:
            eg_input = mod(eg_input)
        feature_extractor_modules.extend([
            self.activation_constructor(),
            nn.Conv1d(base_channels, base_channels, groups=base_channels, kernel_size=eg_input.size(2)),
            nn.Conv1d(base_channels, base_channels, kernel_size=1),
            self.activation_constructor()
        ])
        self.feature_extractor = nn.Sequential(*feature_extractor_modules)
        
        self.classifier_heads = nn.ModuleDict({
            head_key: nn.Linear(base_channels, head_size)
            for head_key, head_size in head_sizes.items()
        })
        
    def forward(self, x):
        x = self.input_transform(x)
        x = self.feature_extractor(x)
        x = x.view(x.size(0), x.size(1))
        out = {head_key: head(x) for head_key, head in self.classifier_heads.items()}
        return out
    
class Discriminator(ModelBase):
    def __init__(self, input_shape, head_sizes, base_channels=16, max_channels=128,
                 ds_blocks=6, iso_blocks=0, kernel_size=3, activation='leaky_relu',
                 embedding_dim=128, norm_layer='instance_norm', block='stargan', use_sn=True):
        super().__init__(activation=activation, embedding_dim=embedding_dim,
                         norm_layer=norm_layer, use_sn=use_sn, block=block)
        
        self.get_ch = lambda n: min((base_channels*2**n, max_channels))
        self.head_sizes = head_sizes
        apply_sn_fn = spectral_norm if use_sn else lambda x: x
        self.input_transform = apply_sn_fn(nn.Conv1d(input_shape[0], base_channels, kernel_size=1, stride=1, padding=0))
        self.downsample_blocks = nn.ModuleList([
            self.block_constructor(
                self.get_ch(n), self.get_ch(n+1), kernel_size=kernel_size,
                activation=self.activation_constructor, resample_layer='downsample',
                norm_layer=self.norm_layer_constructor, use_sn=use_sn
            ) for n in range(ds_blocks)
        ])
        self.iso_blocks = nn.ModuleList([
            self.block_constructor(
                self.get_ch(ds_blocks), self.get_ch(ds_blocks), kernel_size=kernel_size,
                activation=self.activation_constructor, resample_layer='none',
                norm_layer=self.norm_layer_constructor, use_sn=use_sn
            ) for _ in range(iso_blocks)
        ])
        eg_input = torch.randn(1, *input_shape)
        eg_input = self.input_transform(eg_input)
        for mod in self.downsample_blocks:
            eg_input = mod(eg_input)
        for mod in self.iso_blocks:
            eg_input = mod(eg_input)
        self.output_transform = nn.Sequential(
            self.activation_constructor(),
            apply_sn_fn(nn.Conv1d(self.get_ch(ds_blocks), self.get_ch(ds_blocks), groups=self.get_ch(ds_blocks), kernel_size=eg_input.size(-1))),
            apply_sn_fn(nn.Conv1d(self.get_ch(ds_blocks), self.get_ch(ds_blocks), kernel_size=1)),
            self.activation_constructor()
        )
        self.classifier_heads = nn.ModuleDict({
            head_key: apply_sn_fn(nn.Linear(self.get_ch(ds_blocks), head_size))
            for head_key, head_size in head_sizes.items()
        })
        self.class_embedding = apply_sn_fn(nn.Linear(sum(head_sizes.values()), embedding_dim))
        self.realism_head = apply_sn_fn(nn.Linear(embedding_dim, 1))
        
    def extract_features(self, x):
        x = self.input_transform(x)
        for ds_block in self.downsample_blocks:
            x = ds_block(x)
        for iso_block in self.iso_blocks:
            x = iso_block(x)
        x = self.output_transform(x)
        x = x.view(x.size(0), x.size(1))
        return x
    
    def classify_leakage(self, x):
        out = {head_key: head(x) for head_key, head in self.classifier_heads.items()}
        return out
    
    def classify_realism(self, x, y):
        y = deepcopy(y)
        for (y_name, y_val), head_size in zip(y.items(), self.head_sizes.values()):
            if len(y_val.shape) < 2:
                y[y_name] = nn.functional.one_hot(y_val, head_size)
        y = torch.cat(list(y.values()), dim=-1).to(torch.float)
        embedded_y = self.class_embedding(y)
        out = self.realism_head(x) + (x*embedded_y).sum(dim=-1, keepdim=True)
        return out