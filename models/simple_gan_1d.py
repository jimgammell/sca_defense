import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

class LeNet5Generator1d(nn.Module):
    def __init__(self,
                 latent_dims,
                 label_dims,
                 output_shape,
                 feature_dims=0,
                 output_classes=256,
                 apply_spectral_norm=False,
                 apply_batch_norm=True):
        super().__init__()
        
        self.latent_dims = latent_dims
        self.label_dims = label_dims
        self.feature_dims = feature_dims
        self.output_shape = output_shape
        self.output_classes = output_classes
        
        eg_input = torch.randn(output_shape)
        self.feature_encoder = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(in_channels=output_shape[1],
                      out_channels=feature_dims//4,
                      kernel_size=5,
                      padding=2,
                      stride=5,
                      bias=False),
            nn.BatchNorm1d(feature_dims//4),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dims//4,
                      out_channels=feature_dims//2,
                      kernel_size=5,
                      padding=2,
                      stride=5,
                      bias=False),
            nn.BatchNorm1d(feature_dims//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dims//2,
                      out_channels=feature_dims,
                      kernel_size=5,
                      padding=2,
                      stride=5))
        eg_input = self.feature_encoder(eg_input)
        feature_kernel_len = int(np.prod(eg_input.shape[2:]))
        self.feature_kernel_len = feature_kernel_len
        
        self.latent_encoder = nn.Sequential(
            nn.Conv1d(in_channels=latent_dims,
                      out_channels=8*latent_dims,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(8*latent_dims),
            nn.ReLU(),
            nn.Conv1d(in_channels=8*latent_dims,
                      out_channels=latent_dims,
                      kernel_size=1),
            nn.ConvTranspose1d(in_channels=latent_dims,
                               out_channels=latent_dims,
                               kernel_size=feature_kernel_len))
        
        self.label_embedding = nn.Embedding(output_classes, label_dims)
        self.label_encoder = nn.Sequential(
            nn.Conv1d(in_channels=label_dims,
                      out_channels=8*label_dims,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(8*label_dims),
            nn.ReLU(),
            nn.Conv1d(in_channels=8*label_dims,
                      out_channels=label_dims,
                      kernel_size=1),
            nn.ConvTranspose1d(in_channels=label_dims,
                               out_channels=label_dims,
                               kernel_size=feature_kernel_len))
        
        self.output_decoder = nn.Sequential(
            nn.Conv1d(in_channels=feature_dims+latent_dims+label_dims,
                      out_channels=8*feature_dims,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(8*feature_dims),
            nn.ReLU(),
            nn.Conv1d(in_channels=8*feature_dims,
                      out_channels=feature_dims,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(feature_dims),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=feature_dims,
                               out_channels=feature_dims//2,
                               kernel_size=5,
                               stride=5,
                               bias=False),
            nn.BatchNorm1d(feature_dims//2),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=feature_dims//2,
                               out_channels=feature_dims//4,
                               kernel_size=5,
                               stride=5,
                               bias=False),
            nn.BatchNorm1d(feature_dims//4),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=feature_dims//4,
                               out_channels=feature_dims//8,
                               kernel_size=5,
                               stride=5,
                               bias=False),
            nn.BatchNorm1d(feature_dims//8),
            nn.ReLU(),
            nn.Conv1d(in_channels=feature_dims//8,
                      out_channels=output_shape[1],
                      kernel_size=1),
            nn.Upsample(scale_factor=4))
        
    def forward(self, latent, labels, image):
        latent = latent[:image.size(0)]
        encoded_features = self.feature_encoder(image)
        embedded_labels = self.label_embedding(labels).view(-1, self.label_dims, 1)
        encoded_labels = self.label_encoder(embedded_labels)
        encoded_latent = self.latent_encoder(latent.view(-1, self.latent_dims, 1))
        z = torch.cat((encoded_features, encoded_labels, encoded_latent), dim=1)
        generated_image = self.output_decoder(z)
        return generated_image

class LeNet5Discriminator1d(nn.Module):
    def __init__(self,
                 input_shape,
                 output_classes=256,
                 feature_dims=16,
                 apply_spectral_norm=True,
                 apply_batch_norm=True):
        super().__init__()
        
        self.output_transform = nn.Identity()
        self.feature_dims = feature_dims
        
        eg_input = torch.randn(input_shape)
        self.feature_encoder = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            spectral_norm(nn.Conv1d(
                in_channels=input_shape[1],
                out_channels=feature_dims//8,
                kernel_size=5,
                stride=3,
                bias=False)),
            nn.BatchNorm1d(feature_dims//8),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv1d(
                in_channels=feature_dims//8,
                out_channels=feature_dims//4,
                kernel_size=5,
                stride=3,
                bias=False)),
            nn.BatchNorm1d(feature_dims//4),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv1d(
                in_channels=feature_dims//4,
                out_channels=feature_dims//2,
                kernel_size=5,
                stride=3)),
            nn.BatchNorm1d(feature_dims//2),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv1d(
                in_channels=feature_dims//2,
                out_channels=feature_dims,
                kernel_size=5,
                stride=3))
        )
        eg_input = self.feature_encoder(eg_input)
        feature_kernel_len = int(np.prod(eg_input.shape[2:]))
        self.feature_kernel_len = feature_kernel_len
        
        self.mlp_probe = nn.Sequential(
            spectral_norm(nn.Linear(feature_kernel_len*feature_dims, 256)),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 256)),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, output_classes)))
    
    def forward(self, x):
        features = self.feature_encoder(x).view(-1, self.feature_kernel_len*self.feature_dims)
        logits = self.mlp_probe(features)
        return logits
    
    def logits(self, x):
        return self.forward(x)