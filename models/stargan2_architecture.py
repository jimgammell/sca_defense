from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

# Based on this implementation:
#   https://github.com/clovaai/stargan-v2/blob/master/core/model.py
class AdaptiveInstanceNorm1d(nn.InstanceNorm1d):
    def __init__(self, embedding_dim, num_features, *args, use_sn=False, **kwargs):
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
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 embedding_dim=None,
                 activation=lambda: 'relu',
                 use_sn=False,
                 resample_layer='none', norm_layer='none'):
        
        if norm_layer == 'adaptive_instance_norm':
            assert embedding_dim is not None
        activation_constructor = {
            'relu':       lambda: nn.ReLU(inplace=True),
            'leaky_relu': lambda: nn.LeakyReLU(0.1)
        }[activation]
        norm_layer_constructor = {
            'none':                   None,
            'batch_norm':             lambda features: SnBatchNorm1d(features, affine=True, use_sn=use_sn),
            'instance_norm':          lambda features: SnInstanceNorm1d(features, affine=True, use_sn=use_sn),
            'adaptive_instance_norm': lambda features: AdaptiveInstanceNorm1d(embedding_dim, features, use_sn=use_sn)
        }[norm_layer]
        resample_layer_constructor = {
            'none':      None,
            'avg_pool':  lambda: nn.AvgPool1d(2),
            'nn_interp': lambda: nn.Upsample(scale_factor=2, mode='nearest')
        }[resample_layer]
        
        super().__init__()
        
        self.residual_modules = nn.ModuleList([])
        if norm_layer_constructor is not None:
            self.residual_modules.append(norm_layer_constructor(in_channels))
        self.residual_modules.append(activation_constructor())
        self.residual_modules.append(nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
        if resample_layer_constructor is not None:
            self.residual_modules.append(resample_layer_constructor())
        if norm_layer_constructor is not None:
            self.residual_modules.append(norm_layer_constructor(in_channels))
        self.residual_modules.append(activation_constructor())
        self.residual_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        
        self.shortcut_modules = nn.ModuleList([])
        if in_channels != out_channels:
            self.shortcut_modules.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
        if resample_layer_constructor is not None:
            self.shortcut_modules.append(resample_layer_constructor())
        
        if use_sn:
            for mod_idx, mod in enumerate(self.residual_modules):
                if isinstance(mod, nn.Conv1d):
                    self.residual_modules[mod_idx] = spectral_norm(mod)
            for mod_idx, mod in enumerate(self.shortcut_modules):
                if isinstance(mod, nn.Conv1d):
                    self.shortcut_modules[mod_idx] = spectral_norm(mod)
        
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
        
        out = (x_resid + x_sc) / np.sqrt(2)
        return out

class Generator(nn.Module):
    def __init__(self, input_shape, head_sizes, use_sn=True, embedding_dim=256, base_channels=16):
        super().__init__()
        
        apply_sn_fn = spectral_norm if use_sn else lambda x: x
        self.head_sizes = head_sizes
        self.input_transform = apply_sn_fn(nn.Conv1d(in_channels=input_shape[0], out_channels=base_channels, kernel_size=1, stride=1, padding=0))
        downsample_kwargs = {'activation': 'relu', 'resample_layer': 'avg_pool', 'norm_layer': 'instance_norm', 'use_sn': use_sn}
        self.downsample_blocks = nn.ModuleList([
            ResidualBlock(1*base_channels, 2*base_channels,  **downsample_kwargs),
            ResidualBlock(2*base_channels, 4*base_channels, **downsample_kwargs),
            ResidualBlock(4*base_channels, 8*base_channels, **downsample_kwargs),
            ResidualBlock(8*base_channels, 8*base_channels, **downsample_kwargs)
        ])
        pre_endo_kwargs = {'activation': 'relu', 'resample_layer': 'none', 'norm_layer': 'instance_norm', 'use_sn': use_sn}
        self.pre_endo_blocks = nn.ModuleList([
            ResidualBlock(8*base_channels, 8*base_channels, **pre_endo_kwargs),
            ResidualBlock(8*base_channels, 8*base_channels, **pre_endo_kwargs)
        ])
        post_endo_kwargs = {
            'activation': 'relu', 'resample_layer': 'none', 'use_sn': use_sn,
            'norm_layer': 'adaptive_instance_norm', 'embedding_dim': embedding_dim
        }
        self.post_endo_blocks = nn.ModuleList([
            ResidualBlock(8*base_channels, 8*base_channels, **post_endo_kwargs),
            ResidualBlock(8*base_channels, 8*base_channels, **post_endo_kwargs)
        ])
        upsample_kwargs = {
            'activation': 'relu', 'resample_layer': 'nn_interp', 'use_sn': use_sn,
            'norm_layer': 'adaptive_instance_norm', 'embedding_dim': embedding_dim
        }
        self.upsample_blocks = nn.ModuleList([
            ResidualBlock(8*base_channels, 8*base_channels, **upsample_kwargs),
            ResidualBlock(8*base_channels, 4*base_channels, **upsample_kwargs),
            ResidualBlock(4*base_channels, 2*base_channels, **upsample_kwargs),
            ResidualBlock(2*base_channels, 1*base_channels,  **upsample_kwargs)
        ])
        self.output_transform = apply_sn_fn(nn.Conv1d(in_channels=base_channels, out_channels=input_shape[0], kernel_size=1, stride=1, padding=0))
        self.class_embedding = apply_sn_fn(nn.Linear(sum(head_sizes.values()), embedding_dim))
        
    def forward(self, x, y):
        y = deepcopy(y)
        for (y_name, y_val), head_size in zip(y.items(), self.head_sizes.values()):
            if len(y_val.shape) < 2:
                y[y_name] = nn.functional.one_hot(y_val, head_size)
        y = torch.cat(list(y.values()), dim=-1).to(torch.float)
        embedded_y = self.class_embedding(y)
        x = self.input_transform(x)
        x_lengths = []
        for ds_block in self.downsample_blocks:
            x = ds_block(x)
            x_lengths.append(x.size(-1))
        for pr_block in self.pre_endo_blocks:
            x = pr_block(x)
        for po_block in self.post_endo_blocks:
            x = po_block(x, embedded_y)
        for us_block, l in zip(self.upsample_blocks, x_lengths[::-1]):
            if x.size(2) < l:
                pad_amt = l - x.size(2)
                x = nn.functional.pad(x, (pad_amt//2, (pad_amt//2)+(1 if pad_amt%2!=0 else 0)))
            x = us_block(x, embedded_y)
        x = self.output_transform(x)
        x = torch.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_shape, head_sizes, base_channels=16, use_sn=True):
        super().__init__()
        
        apply_sn_fn = spectral_norm if use_sn else lambda x: x
        self.head_sizes = head_sizes
        self.input_transform = apply_sn_fn(nn.Conv1d(1, base_channels, kernel_size=input_shape[0], stride=1, padding=0))
        ds_kwargs = {'activation': 'leaky_relu', 'resample_layer': 'avg_pool', 'norm_layer': 'instance_norm', 'use_sn': use_sn}
        self.downsample_blocks = nn.ModuleList([
            ResidualBlock(1*base_channels, 2*base_channels, **ds_kwargs),
            ResidualBlock(2*base_channels, 4*base_channels, **ds_kwargs),
            ResidualBlock(4*base_channels, 8*base_channels, **ds_kwargs),
            ResidualBlock(8*base_channels, 8*base_channels, **ds_kwargs),
            ResidualBlock(8*base_channels, 8*base_channels, **ds_kwargs),
            ResidualBlock(8*base_channels, 8*base_channels, **ds_kwargs)
        ])
        remaining_dims = input_shape[-1]
        for _ in range(6):
            remaining_dims = remaining_dims // 2
        pre_pool_kernel_size = int(np.sqrt(remaining_dims))
        self.output_transform = nn.Sequential(
            nn.LeakyReLU(0.1),
            apply_sn_fn(nn.Conv1d(8*base_channels, 8*base_channels, groups=8*base_channels, kernel_size=remaining_dims, stride=1, padding=0)),
            apply_sn_fn(nn.Conv1d(8*base_channels, 8*base_channels, kernel_size=1, stride=1, padding=0)),
            nn.LeakyReLU(0.1)
        )
        self.classifier_heads = nn.ModuleDict({
            head_key: apply_sn_fn(nn.Linear(8*base_channels, head_size))
            for head_key, head_size in head_sizes.items()
        })
        embedding_dim = 8*base_channels
        self.class_embedding = apply_sn_fn(nn.Linear(sum(head_sizes.values()), embedding_dim))
        self.realism_head = apply_sn_fn(nn.Linear(embedding_dim, 1))
        
    def extract_features(self, x):
        x = self.input_transform(x)
        for ds_block in self.downsample_blocks:
            x = ds_block(x)
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