# Based on this tutorial: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

import torch
from torch import nn
from torch.nn.utils import spectral_norm

class DCGenerator(nn.Module):
    def __init__(self,
                 latent_dims,
                 label_dims,
                 output_shape,
                 feature_maps=32,
                 output_transform=nn.Hardtanh):
        self.latent_dims = latent_dims
        self.label_dims = label_dims
        self.output_shape = output_shape
        if self.label_dims > 0:
            self.use_labels = True
            n_inputs = latent_dims + label_dims
        else:
            self.use_labels = False
            n_inputs = latent_dims
        self.n_inputs = n_inputs
        
        super().__init__()
        self.feature_creator = nn.Sequential(
            nn.ConvTranspose2d(n_inputs, feature_maps*8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps*8),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps*8, feature_maps*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps*4, feature_maps*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps*2, feature_maps, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(),
            nn.ConvTranspose2d(feature_maps, output_shape[1], 4, stride=2, padding=1, bias=False)
        )
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
            z = torch.cat((latent_vars, embedded_labels), dim=-1).view(-1, self.n_inputs, 1, 1)
        else:
            (latent_vars,) = args
            z = latent_vars.view(-1, self.n_inputs, 1, 1)
        logits = self.feature_creator(z)
        image = nn.functional.interpolate(logits, size=self.output_shape[2:])
        image = self.output_transform(image)
        return image
    
class DCDiscriminator(nn.Module):
    def __init__(self,
                 input_shape,
                 output_classes=10,
                 feature_maps=32,
                 apply_spectral_norm=True):
        super().__init__()
        eg_input = torch.rand(input_shape)
        f = spectral_norm if apply_spectral_norm else lambda x: x
        self.feature_extractor = nn.Sequential(
            f(nn.Conv2d(input_shape[1], feature_maps, 4, stride=1, padding=2, bias=False)),
            nn.LeakyReLU(0.2),
            f(nn.Conv2d(feature_maps, 2*feature_maps, 4, stride=1, padding=1, bias=False)),
            nn.BatchNorm2d(2*feature_maps),
            nn.LeakyReLU(0.2),
            f(nn.Conv2d(2*feature_maps, 4*feature_maps, 4, stride=1, padding=2, bias=False)),
            nn.BatchNorm2d(4*feature_maps),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4*feature_maps, output_classes, 4, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )
        kernel_size = self.feature_extractor(eg_input).shape[2:]
        self.pooling_layer = nn.AvgPool2d(kernel_size)
        self.output_transform = nn.Identity()
        self.output_classes = output_classes
        
    def logits(self, x):
        features = self.feature_extractor(x)
        logits = self.pooling_layer(features).view(-1, self.output_classes)
        return logits
    
    def forward(self, x):
        logits = self.logits(x)
        output = self.output_transform(logits)
        return output