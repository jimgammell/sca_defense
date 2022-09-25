import numpy as np
import torch
from torch import nn

from models.common import get_param_count

# Batch norm with the default Tensorflow arguments.
class BatchNorm1d(nn.BatchNorm1d):
    def __init(self, *args, **kwargs):
        if not 'momentum' in kwargs.keys():
            kwargs['momentum'] = 0.01
        if not 'eps' in kwargs.keys():
            kwargs['eps'] = 1e-3
        super().__init__(*args, **kwargs)

class ResNet1dGenerator(nn.Module):        
    class Stack(nn.Module):
        class Block(nn.Module):
            def __init__(self,
                         eg_input,
                         filters,
                         kernel_size=3,
                         strides=1,
                         conv_shortcut=False,
                         activation=nn.ReLU(),
                         activation_kwargs={}):
                super().__init__()
            
                self.input_shape = eg_input.shape
                self.filters = filters
                self.kernel_size = kernel_size
                self.strides = strides
                self.conv_shortcut = False
                self.activation = nn.ReLU
                self.activation_kwargs = activation_kwargs
                def Activation():
                    return self.activation(**self.activation_kwargs)

                self.input_transform = nn.Sequential(
                    BatchNorm1d(num_features=self.input_shape[1]),
                    Activation())
                eg_input = self.input_transform(eg_input)

                self.shortcut = nn.Sequential(
                    nn.ConvTranspose1d(in_channels=self.input_shape[1],
                                       out_channels=self.filters,
                                       kernel_size=1,
                                       stride=self.strides),
                    nn.ConstantPad1d((0, self.strides//2), 0))

                self.residual = nn.Sequential(
                    nn.ConvTranspose1d(in_channels=self.input_shape[1],
                                       out_channels=4*self.filters,
                                       kernel_size=1,
                                       bias=False),
                    BatchNorm1d(num_features=4*self.filters),
                    Activation(),
                    nn.ConvTranspose1d(in_channels=4*self.filters,
                                       out_channels=4*self.filters,
                                       kernel_size=self.kernel_size,
                                       stride=self.strides,
                                       bias=False),
                    nn.ConstantPad1d((0, self.strides//2), 0),
                    BatchNorm1d(num_features=4*self.filters),
                    Activation(),
                    nn.ConvTranspose1d(in_channels=4*self.filters,
                                       out_channels=self.filters,
                                       kernel_size=1))

            def forward(self, x):
                transformed_x = self.input_transform(x)
                id_x = self.shortcut(transformed_x)
                resid_x = self.residual(transformed_x)
                resid_x = resid_x[:, :, self.kernel_size//2:-(self.kernel_size//2)] # transpose of 'same' padding
                logits = id_x + resid_x
                return logits
        
        def __init__(self,
                     eg_input,
                     filters,
                     blocks,
                     kernel_size=3,
                     strides=2,
                     activation=nn.ReLU,
                     activation_kwargs={}):
            super().__init__()
            
            self.input_shape = eg_input.shape
            self.filters = filters
            self.blocks = blocks
            self.kernel_size = kernel_size
            self.strides = strides
            self.activation = activation
            self.activation_kwargs = activation_kwargs
            def Block(eg_input, kernel_size=None, strides=None, conv_shortcut=None):
                kwargs = {'filters': self.filters,
                          'activation': self.activation,
                          'activation_kwargs': self.activation_kwargs}
                if kernel_size is not None:
                    kwargs.update({'kernel_size': kernel_size})
                if strides is not None:
                    kwargs.update({'strides': strides})
                if conv_shortcut is not None:
                    kwargs.update({'conv_shortcut': conv_shortcut})
                return self.Block(eg_input, **kwargs)
            
            modules = [Block(eg_input,
                             kernel_size=self.kernel_size,
                             conv_shortcut=True)]
            for _ in range(2, self.blocks):
                eg_input = modules[-1](eg_input)
                modules.append(Block(eg_input,
                                     kernel_size=self.kernel_size))
            eg_input = modules[-1](eg_input)
            modules.append(Block(eg_input,
                                 strides=strides))
            self.model = nn.Sequential(*modules)
        
        def forward(self, x):
            logits = self.model(x)
            return logits
    
    class FeatureEncoder(nn.Module):
        def __init__(self,
                     eg_input,
                     feature_dims,
                     kernel_size,
                     pool_size):
            super().__init__()
            
            self.input_downsampler = nn.MaxPool1d(kernel_size=pool_size)
            feature_extractors = []
            shortcuts = []
            prev_features = 1
            current_features = 8
            while current_features <= 128:
                feature_extractors.append(nn.Sequential(
                    nn.Conv1d(in_channels=prev_features,
                              out_channels=4*current_features,
                              kernel_size=1,
                              bias=False),
                    nn.BatchNorm1d(num_features=4*current_features),
                    nn.ReLU(),
                    nn.ConstantPad1d(padding=(kernel_size//2, kernel_size//2), value=0),
                    nn.Conv1d(in_channels=4*current_features,
                              out_channels=4*current_features,
                              kernel_size=kernel_size,
                              stride=2,
                              bias=False),
                    nn.BatchNorm1d(num_features=4*current_features),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=4*current_features,
                              out_channels=current_features,
                              kernel_size=1,
                              bias=False),
                    nn.BatchNorm1d(num_features=current_features),
                    nn.ReLU()))
                shortcuts.append(nn.Conv1d(in_channels=prev_features,
                                           out_channels=current_features,
                                           kernel_size=1,
                                           stride=2))
                prev_features = current_features
                current_features *= 2
            
            self.feature_extractors = nn.ModuleList(feature_extractors)
            self.shortcuts = nn.ModuleList(shortcuts)
            
            eg_input = self.input_downsampler(eg_input)
            for feature_extractor, shortcut in zip(self.feature_extractors, self.shortcuts):
                eg_input = feature_extractor(eg_input) + shortcut(eg_input)
            self.output_transform = nn.Sequential(nn.Conv1d(in_channels=prev_features,
                                                            out_channels=feature_dims,
                                                            kernel_size=1),
                                                  nn.AvgPool1d(kernel_size=eg_input.shape[-1]))
            
        def forward(self, x):
            x = self.input_downsampler(x)
            for feature_extractor, shortcut in zip(self.feature_extractors, self.shortcuts):
                x = feature_extractor(x) + shortcut(x)
            x = self.output_transform(x)
            return x
        
    def __init__(self,
                 latent_dims,
                 label_dims,
                 output_shape,
                 feature_dims=100,
                 output_transform=nn.Hardtanh,
                 pool_size=4,
                 filters=8,
                 block_kernel_size=3,
                 activation=nn.ReLU,
                 activation_kwargs={},
                 dense_dropout=0.1,
                 stack_sizes=[3, 4, 4, 3]):
        super().__init__()
        
        self.label_dims = label_dims
        self.latent_dims = latent_dims
        self.feature_dims = feature_dims
        self.output_shape = output_shape
        self.output_transform = output_transform()
        self.pool_size = pool_size
        self.filters = filters
        self.block_kernel_size = block_kernel_size
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        self.dense_dropout = dense_dropout
        self.stack_sizes = stack_sizes
        
        self.feature_encoder = self.FeatureEncoder(torch.randn(output_shape), feature_dims, block_kernel_size, pool_size)
        
        eg_label = torch.randint(0, self.label_dims, (2,))
        eg_latent = torch.randn(2, self.latent_dims)
        eg_trace = torch.randn(2, *self.output_shape[1:])
        self.label_embedding = nn.Embedding(self.label_dims, self.label_dims)
        eg_embedded_label = self.label_embedding(eg_label)
        eg_embedded_label = eg_embedded_label.view(-1, self.label_dims)
        eg_embedded_trace = self.feature_encoder(eg_trace).view(eg_trace.shape[0], -1)
        eg_z = torch.cat((eg_embedded_label, eg_latent, eg_embedded_trace), dim=-1).view(-1, self.label_dims+self.latent_dims+self.feature_dims, 1)
        
        upscaling_modules = [
            nn.ConvTranspose1d(in_channels=self.label_dims+self.latent_dims+self.feature_dims,
                               out_channels=self.label_dims+self.latent_dims+self.feature_dims,
                               kernel_size=1,
                               bias=False),
            BatchNorm1d(num_features=self.label_dims+self.latent_dims+self.feature_dims),
            self.activation(**self.activation_kwargs),
            nn.ConvTranspose1d(in_channels=self.label_dims+self.latent_dims+self.feature_dims,
                               out_channels=self.label_dims+self.latent_dims+self.feature_dims,
                               kernel_size=np.prod(self.output_shape[1:])//(125*2**len(self.stack_sizes))),
            nn.Upsample(scale_factor=125),
            nn.Dropout(self.dense_dropout)]
        self.input_upscaling = nn.Sequential(*upscaling_modules)
        eg_input = self.input_upscaling(eg_z)
        
        modules = []
        filters *= 2**len(self.stack_sizes)
        for stack_idx, stack_size in enumerate(self.stack_sizes):
            filters = filters//2
            modules.append(self.Stack(eg_input,
                                      filters,
                                      stack_size,
                                      kernel_size=block_kernel_size,
                                      activation=self.activation,
                                      activation_kwargs=self.activation_kwargs))
            eg_input = modules[-1](eg_input)
        modules.append(nn.ConvTranspose1d(in_channels=filters,
                                          out_channels=1,
                                          kernel_size=1))
        self.feature_creator = nn.Sequential(*modules)
        
    def forward(self, latent_vars, labels, traces):
        latent_vars = latent_vars[:labels.size(0), :]
        labels = self.label_embedding(labels)
        trace_features = self.feature_encoder(traces).view(traces.size(0), -1)
        z = torch.cat((labels, latent_vars, trace_features), dim=-1).view(-1, self.label_dims+self.latent_dims+self.feature_dims, 1)
        upscaled_input = self.input_upscaling(z)
        logits = self.feature_creator(upscaled_input)
        if logits.shape[1:] != self.output_shape[1:]:
            logits = nn.functional.interpolate(logits, size=self.output_shape[2:])
        image = self.output_transform(logits)
        return image

class ResNet1dDiscriminator(nn.Module):
    class Stack(nn.Module):
        class Block(nn.Module):
            def __init__(self,
                         eg_input,
                         filters,
                         kernel_size=3,
                         strides=1,
                         conv_shortcut=False,
                         activation=nn.LeakyReLU,
                         activation_kwargs={'negative_slope': 0.2},
                         spectral_norm=False):
                super().__init__()

                self.input_shape = eg_input.shape
                self.filters = filters
                self.kernel_size = kernel_size
                self.strides = strides
                self.conv_shortcut = conv_shortcut
                self.activation = activation
                self.activation_kwargs = activation_kwargs
                self.spectral_norm = spectral_norm
                def Conv1d(*args, **kwargs):
                    if self.spectral_norm:
                        return torch.nn.utils.spectral_norm(nn.Conv1d(*args, **kwargs))
                    else:
                        return nn.Conv1d(*args, **kwargs)
                def Activation():
                    return self.activation(**self.activation_kwargs)

                self.input_transform = nn.Sequential(
                    BatchNorm1d(num_features=self.input_shape[1]),
                    Activation())
                eg_input = self.input_transform(eg_input)

                if self.conv_shortcut:
                    self.shortcut = Conv1d(in_channels=self.input_shape[1],
                                           out_channels=4*self.filters,
                                           kernel_size=1,
                                           stride=self.strides)
                else:
                    if self.strides > 1:
                        self.shortcut = nn.MaxPool1d(kernel_size=1,
                                                     stride=self.strides)
                    else:
                        self.shortcut = nn.Identity()

                self.residual = nn.Sequential(
                    Conv1d(in_channels=self.input_shape[1],
                           out_channels=self.filters,
                           kernel_size=1,
                           bias=False),
                    BatchNorm1d(num_features=self.filters),
                    Activation(),
                    nn.ConstantPad1d(padding=(self.kernel_size//2, self.kernel_size//2),
                                     value=0),
                    Conv1d(in_channels=self.filters,
                           out_channels=self.filters,
                           kernel_size=self.kernel_size,
                           stride=self.strides,
                           bias=False),
                    BatchNorm1d(num_features=self.filters),
                    Activation(),
                    Conv1d(in_channels=self.filters,
                           out_channels=4*self.filters,
                           kernel_size=1))

            def forward(self, x):
                transformed_x = self.input_transform(x)
                id_x = self.shortcut(transformed_x)
                resid_x = self.residual(transformed_x)
                logits = id_x + resid_x
                return logits
        
        def __init__(self,
                     eg_input,
                     filters,
                     blocks,
                     kernel_size=3,
                     strides=2,
                     activation=nn.LeakyReLU,
                     activation_kwargs={'negative_slope': 0.2},
                     spectral_norm=False):
            super().__init__()
            
            self.input_shape = eg_input.shape
            self.filters = filters
            self.blocks = blocks
            self.kernel_size = kernel_size
            self.strides = strides
            self.activation = activation
            self.activation_kwargs = activation_kwargs
            self.spectral_norm = spectral_norm
            def Block(eg_input, kernel_size=None, strides=None, conv_shortcut=None):
                kwargs = {'filters': self.filters,
                          'activation': self.activation,
                          'activation_kwargs': self.activation_kwargs,
                          'spectral_norm': self.spectral_norm}
                if kernel_size is not None:
                    kwargs.update({'kernel_size': kernel_size})
                if strides is not None:
                    kwargs.update({'strides': strides})
                if conv_shortcut is not None:
                    kwargs.update({'conv_shortcut': conv_shortcut})
                return self.Block(eg_input, **kwargs)
            
            modules = [Block(eg_input,
                             kernel_size=self.kernel_size,
                             conv_shortcut=True)]
            for _ in range(2, self.blocks):
                eg_input = modules[-1](eg_input)
                modules.append(Block(eg_input,
                                     kernel_size=self.kernel_size))
            eg_input = modules[-1](eg_input)
            modules.append(Block(eg_input,
                                 strides=strides))
            self.model = nn.Sequential(*modules)
            
        def forward(self, x):
            logits = self.model(x)
            return logits
    
    def __init__(self,
                 input_shape,
                 pool_size=4,
                 filters=8,
                 block_kernel_size=3,
                 activation=nn.LeakyReLU,
                 activation_kwargs={'negative_slope': 0.2},
                 apply_spectral_norm=False,
                 dense_dropout=0.1,
                 stack_sizes=[3, 4, 4, 3]):
        super().__init__()
        
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.filters = filters
        self.block_kernel_size = block_kernel_size
        self.activation = activation
        self.activation_kwargs = activation_kwargs
        self.spectral_norm = apply_spectral_norm
        self.dense_dropout = dense_dropout
        self.stack_sizes = stack_sizes
        self.output_transform = nn.Identity()
        
        eg_input = torch.randn(self.input_shape)
        self.input_transform = nn.MaxPool1d(kernel_size=pool_size)
        eg_input = self.input_transform(eg_input)
        
        modules = []
        for stack_idx, stack_size in enumerate(stack_sizes):
            filters *= 2
            modules.append(self.Stack(eg_input,
                                      filters,
                                      stack_size,
                                      kernel_size=block_kernel_size,
                                      activation=self.activation,
                                      activation_kwargs=self.activation_kwargs,
                                      spectral_norm=self.spectral_norm))
            eg_input = modules[-1](eg_input)
        self.feature_extractor = nn.Sequential(*modules)
        
        self.pooling_layer = nn.AvgPool1d(kernel_size=eg_input.shape[-1])
        eg_input = self.pooling_layer(eg_input)
        eg_input = eg_input.view(-1, np.prod(eg_input.shape[1:]))
        
        def Linear(*args, **kwargs):
            if self.spectral_norm:
                return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))
            else:
                return nn.Linear(*args, **kwargs)
        self.dense_probe = nn.Sequential(
            nn.Dropout(self.dense_dropout),
            Linear(eg_input.shape[1], 256),
            BatchNorm1d(num_features=256),
            self.activation(**self.activation_kwargs),
            Linear(256, 256))
        eg_input = self.dense_probe(eg_input)
        
    def forward(self, x):
        transformed_x = self.input_transform(x)
        features = self.feature_extractor(transformed_x)
        pooled_features = self.pooling_layer(features)
        logits = self.dense_probe(pooled_features.view(-1, np.prod(pooled_features.shape[1:])))
        return logits
    def logits(self, x):
        return self.forward(x)
