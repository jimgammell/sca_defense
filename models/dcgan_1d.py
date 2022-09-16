

import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

class DCGenerator1d(nn.Module):
    def __init__(self,
                 latent_dims,
                 label_dims,
                 output_shape,
                 feature_maps=32,
                 output_transform=nn.Hardtanh):
        self.latent_dims = latent_dims
        self.label_dims = label_dims
        self.output_shape = output_shape
        self.feature_maps = feature_maps
        if self.label_dims > 0:
            self.use_labels = True
            n_inputs = latent_dims + label_dims
        else:
            self.use_labels = False
            n_inputs = latent_dims
        self.n_inputs = n_inputs
        eg_input = torch.randn(1, n_inputs, 1)
        
        kernel_size = 8
        stride = 4
        num_layers = int(np.ceil(np.log(np.prod(output_shape[1:])/kernel_size)/np.log(stride)))
        super().__init__()
        feature_creator_modules = []
        feature_creator_modules.extend([
            nn.ConvTranspose1d(n_inputs, feature_maps*(2**num_layers), kernel_size, stride=1, bias=False),
            nn.BatchNorm1d(feature_maps*(2**num_layers)),
            nn.ReLU()])
        while num_layers > 0:
            feature_creator_modules.extend([
                nn.ConvTranspose1d(feature_maps*(2**num_layers), feature_maps*(2**(num_layers-1)), kernel_size, stride=stride, padding=stride//2, bias=False),
                nn.BatchNorm1d(feature_maps*(2**(num_layers-1))),
                nn.ReLU()])
            num_layers -= 1
        feature_creator_modules.extend([
            nn.ConvTranspose1d(feature_maps, output_shape[1], kernel_size, stride=1, bias=False)])
        self.feature_creator = nn.Sequential(*feature_creator_modules)
        if self.use_labels:
            self.label_embedding = nn.Embedding(label_dims, label_dims)
        if type(output_transform) == str:
            output_transform = getattr(nn, output_transform)
        self.output_transform = output_transform()
        
    def forward(self, *args):
        if self.use_labels:
            (latent_vars, labels) = args
            latent_vars = latent_vars[:labels.size(0), :]
            embedded_labels = self.label_embedding(labels).view(-1, self.label_dims)
            z = torch.cat((latent_vars, embedded_labels), dim=-1).view(-1, self.n_inputs, 1)
        else:
            (latent_vars,) = args
            z = latent_vars.view(-1, self.n_inputs, 1)
        logits = self.feature_creator(z)
        image = nn.functional.interpolate(logits, size=self.output_shape[2:])
        image = self.output_transform(image)
        return image

class DCDiscriminator1d(nn.Module):
    def __init__(self,
                 input_shape,
                 output_classes=10,
                 feature_maps=256,
                 n_layers=5,
                 apply_spectral_norm=True):
        super().__init__()
        eg_input = torch.randn(input_shape)
        kernel_size = 8
        f = spectral_norm if apply_spectral_norm else lambda x: x
        feature_extractor_modules = []
        feature_extractor_modules.extend([
            f(nn.Conv1d(input_shape[1], feature_maps, kernel_size, stride=4, bias=False)),
            nn.LeakyReLU(0.2)])
        for _ in range(n_layers-2):
            feature_extractor_modules.extend([
                f(nn.Conv1d(feature_maps, 2*feature_maps, kernel_size, stride=1, bias=False)),
                nn.BatchNorm1d(2*feature_maps),
                nn.LeakyReLU(0.2)])
            feature_maps *= 2
        feature_extractor_modules.extend([
            f(nn.Conv1d(feature_maps, output_classes, kernel_size, stride=1, bias=False))])
        self.feature_extractor = nn.Sequential(*feature_extractor_modules)
        output_kernel_size = self.feature_extractor(eg_input).shape[-1]
        self.pooling_layer = nn.AvgPool1d(output_kernel_size)
        self.output_classes = output_classes
        
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.pooling_layer(features).view(-1, self.output_classes)
        return logits
        