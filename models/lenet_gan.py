import numpy as np
import torch
from torch import nn

class SimpleGen(nn.Module):
    def __init__(self,
                 latent_dims,
                 label_dims,
                 output_shape,
                 feature_dims=0,
                 output_classes=10,
                 apply_spectral_norm=True,
                 apply_batch_norm=True):
        super().__init__()
        
        self.feature_dims = feature_dims
        
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=8,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=0),
            nn.ReLU())
        
        self.label_encoder = nn.Sequential(
            nn.Embedding(output_classes, output_classes),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(output_classes, 3*3))
        
        latent_channels = latent_dims//(3*3) + int(latent_dims % (3*3) != 0)
        self.latent_channels = latent_channels
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dims, 2*3*3*latent_channels),
            nn.ReLU(),
            nn.Linear(2*3*3*latent_channels, 3*3*latent_channels))
        
        self.feature_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32 + 1 + latent_channels,
                               out_channels=16,
                               kernel_size=3,
                               stride=2,
                               output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,
                               out_channels=8,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8,
                               out_channels=1,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1))
        
    def forward(self, latent, labels, image):
        latent = latent[:image.size(0)]
        encoded_features = self.feature_encoder(image).view(-1, 32, 3, 3)
        encoded_labels = self.label_encoder(labels).view(-1, 1, 3, 3)
        encoded_latent = self.latent_encoder(latent).view(-1, self.latent_channels, 3, 3)
        z = torch.cat((encoded_features, encoded_labels, encoded_latent), dim=1).view(-1, 32+1+self.latent_channels, 3, 3)
        decoded_image = self.feature_decoder(z)
        return decoded_image

class LeNet5Gen(nn.Module):
    def __init__(self,
                 latent_dims,
                 label_dims,
                 output_shape,
                 feature_dims=0,
                 output_classes=10,
                 apply_spectral_norm=True,
                 apply_batch_norm=True):
        super().__init__()
        
        self.latent_dims = latent_dims
        self.label_dims = label_dims
        self.feature_dims = feature_dims
        self.output_shape = output_shape
        
        def get_convtranspose2d(*args, **kwargs):
            if apply_spectral_norm:
                return nn.utils.spectral_norm(nn.ConvTranspose2d(*args, **kwargs))
            else:
                return nn.ConvTranspose2d(*args, **kwargs)
        def get_conv2d(*args, **kwargs):
            if apply_spectral_norm:
                return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
            else:
                return nn.Conv2d(*args, **kwargs)
        def get_linear(*args, **kwargs):
            if apply_spectral_norm:
                return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))
            else:
                return nn.Linear(*args, **kwargs)
        def get_batchnorm2d(*args, **kwargs):
            if apply_batch_norm:
                return nn.BatchNorm2d(*args, **kwargs)
            else:
                return nn.Identity()
            
        if label_dims != 0:
            self.label_embedding = nn.Embedding(output_classes, output_classes)
        
        if feature_dims != 0:
            self.feature_extractor = nn.Sequential(
                nn.Flatten(),
                get_linear(np.prod(output_shape[1:]), 2*feature_dims),
                nn.ReLU(),
                get_linear(2*feature_dims, feature_dims))
            
        self.dense_upsampling = nn.Sequential(
            get_linear(latent_dims + label_dims + feature_dims, 256),
            nn.ReLU(),
            get_linear(256, 512),
            nn.ReLU(),
            get_linear(512, 64*5*5))
        
        self.feature_constructor = nn.Sequential(
            nn.Upsample(scale_factor=2),
            get_batchnorm2d(64),
            nn.ReLU(),
            get_convtranspose2d(in_channels=64,
                                out_channels=32,
                                kernel_size=5,
                                stride=1,
                                padding=0,
                                bias=True),
            nn.Upsample(scale_factor=2),
            get_batchnorm2d(32),
            nn.ReLU(),
            get_convtranspose2d(in_channels=32,
                                out_channels=output_shape[1],
                                kernel_size=5,
                                stride=1,
                                padding=2,
                                bias=True))
        
    def forward(self, *args):
        args_idx = 0
        input_tensors = []
        if self.latent_dims > 0:
            latent_vars = args[args_idx]
            input_tensors.append(latent_vars)
            args_idx += 1
        if self.label_dims > 0:
            labels = args[args_idx]
            embedded_labels = self.label_embedding(labels).view(labels.size(0), -1)
            input_tensors.append(embedded_labels)
            args_idx += 1
        if self.feature_dims > 0:
            sample = args[args_idx]
            features = self.feature_extractor(sample).view(-1, self.feature_dims)
            input_tensors.append(features)
        min_dim = min(t.size(0) for t in input_tensors)
        input_tensors = [t[:min_dim] for t in input_tensors]
        z = torch.cat(input_tensors, dim=-1)
        upsampled_z = self.dense_upsampling(z).view(-1, 64, 5, 5)
        logits = self.feature_constructor(upsampled_z)
        return logits

class LeNet5Disc(nn.Module):
    def __init__(self,
                 input_shape,
                 output_classes=10,
                 apply_spectral_norm=True,
                 apply_batch_norm=True):
        super().__init__()
        
        self.output_transform = nn.Identity()
        
        def get_conv2d(*args, **kwargs):
            if apply_spectral_norm:
                return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))
            else:
                return nn.Conv2d(*args, **kwargs)
        def get_linear(*args, **kwargs):
            if apply_spectral_norm:
                return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))
            else:
                return nn.Linear(*args, **kwargs)
        def get_batchnorm2d(*args, **kwargs):
            if apply_batch_norm:
                return nn.BatchNorm2d(*args, **kwargs)
            else:
                return nn.Identity()
        
        self.feature_extractor = nn.Sequential(
            get_conv2d(in_channels=input_shape[1],
                       out_channels=6,
                       kernel_size=5,
                       stride=1,
                       padding=2,
                       bias=True),
            get_batchnorm2d(6),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            get_conv2d(in_channels=6,
                       out_channels=16,
                       kernel_size=5,
                       stride=1,
                       padding=0,
                       bias=True),
            get_batchnorm2d(16),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.dense_probe = nn.Sequential(
            get_linear(16*5*5, 120),
            nn.LeakyReLU(negative_slope=0.2),
            get_linear(120, 84),
            nn.LeakyReLU(negative_slope=0.2),
            get_linear(84, output_classes))
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.dense_probe(features.view(-1, np.prod(features.shape[1:])))
        return logits
    
    def logits(self, x):
        return self.forward(x)