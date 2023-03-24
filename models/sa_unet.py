import numpy as np
import torch
from torch import nn
from torch.nn.utils import spectral_norm

def init_weights(gain):
    def _init_weights(module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.orthogonal_(module.weight, gain=gain)
    return _init_weights

def get_resample_layer(channels, fixed_resample=True, downsample=False, upsample=False, sn=spectral_norm):
    if downsample:
        assert not upsample
        if fixed_resample:
            resample = nn.AvgPool2d(2)
        else:
            resample = sn(nn.Conv2d(channels, channels, kernel_size=2, stride=2, padding=0))
    elif upsample:
        assert not downsample
        if fixed_resample:
            resample = nn.Upsample(scale_factor=2)
        else:
            resample = sn(nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, padding=0))
    else:
        resample = nn.Identity()
    return resample

# Based on this implementation:
#   https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
class SelfAttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.query_conv = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, width, height = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = nn.functional.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(batch_size, -1, width*height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, width, height)
        out = self.gamma*out + x
        return out

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fixed_resample=False, downsample=False, upsample=False, sn=spectral_norm, activation=lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        
        self.residual_connection = nn.Sequential(
            sn(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            activation(),
            sn(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            activation(),
            get_resample_layer(out_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample, sn=sn)
        )
        self.skip_connection = nn.Sequential(
            sn(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)),
            get_resample_layer(out_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample, sn=sn)
        )
        
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.skip_connection(x)
        out = x_rc + x_sc
        return out

class WideDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fixed_resample=False, downsample=False, upsample=False, sn=spectral_norm, activation=lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        
        self.residual_connection = nn.Sequential(
            sn(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)),
            activation(),
            get_resample_layer(out_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample, sn=sn)
        )
    
    def forward(self, x):
        x_rc = self.residual_connection(x)
        return x_rc

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fixed_resample=True, down_block=True, downsample=False, upsample=False, sn=spectral_norm, activation=nn.ReLU, output_bias=True):
        super().__init__()
        
        if down_block:
            self.residual_connection = nn.Sequential(
                sn(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                activation(),
                nn.BatchNorm2d(out_channels),
                get_resample_layer(out_channels, downsample=downsample, upsample=upsample, sn=sn),
                sn(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            )
        else:
            self.residual_connection = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                activation(),
                get_resample_layer(in_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample, sn=sn),
                sn(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(out_channels),
                activation(),
                sn(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=output_bias))
            )
        self.skip_connection = nn.Sequential(
            get_resample_layer(in_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample, sn=sn),
            sn(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=output_bias))
        )
        
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.skip_connection(x)
        out = x_rc + x_sc
        return out
    
class WideGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, fixed_resample=True, downsample=False, upsample=False, sn=spectral_norm, activation=nn.ReLU, output_bias=True):
        super().__init__()
        
        self.residual_connection = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            activation(),
            get_resample_layer(in_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample, sn=sn),
            sn(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=output_bias))
        )
        self.skip_connection = nn.Sequential(
            get_resample_layer(in_channels, fixed_resample=fixed_resample, downsample=downsample, upsample=upsample, sn=sn),
            sn(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=output_bias))
        )
        
    def forward(self, x):
        x_rc = self.residual_connection(x)
        x_sc = self.skip_connection(x)
        out = x_rc + x_sc
        return out
    
class SubmoduleWrapper(nn.Module):
    def __init__(self, submodule, submodule_channels, num_straight_blocks, sa_block=True, fixed_resample=True, sn=spectral_norm, activation=nn.ReLU, generator_block=GeneratorBlock):
        super().__init__()
        
        self.resample_path = nn.Sequential(
            generator_block(submodule_channels//2, submodule_channels, downsample=True, activation=activation, fixed_resample=fixed_resample),
            submodule,
            generator_block(submodule_channels, submodule_channels//2, upsample=True, activation=activation, fixed_resample=fixed_resample)
        )
        self.skip_path = nn.Sequential(
          *[generator_block(submodule_channels//2, submodule_channels//2, activation=activation, sn=sn, down_block=True) for _ in range(num_straight_blocks//2)],
            SelfAttentionBlock(submodule_channels//2) if sa_block else nn.Identity(),
          *[generator_block(submodule_channels//2, submodule_channels//2, activation=activation, sn=sn, down_block=False) for _ in range(num_straight_blocks//2)]
        )
        self.rejoin = sn(nn.Conv2d(submodule_channels, submodule_channels//2, kernel_size=1, stride=1, padding=0))
        
    def forward(self, x):
        x_rp = self.resample_path(x)
        x_sp = self.skip_path(x)
        out = x_rp + x_sp
        x_comb = torch.cat((x_rp, x_sp), dim=1)
        out = self.rejoin(x_comb)
        return out
    
class Generator(nn.Module):
    def __init__(self, input_shape, initial_channels=8, downsample_blocks=1, sa_block=False, fixed_resample=False, use_sn=True):
        super().__init__()
        
        sn = spectral_norm if use_sn else lambda x: x
        activation = lambda: nn.ReLU(inplace=True)
        generator_block = GeneratorBlock
        
        self.input_transform = generator_block(input_shape[0], initial_channels, sn=sn, activation=activation, down_block=True)
        self.model = nn.Sequential(
            generator_block(initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks, activation=activation, sn=sn, down_block=True),
            SelfAttentionBlock(initial_channels*2**downsample_blocks) if sa_block else nn.Identity(),
            generator_block(initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks, activation=activation, sn=sn, down_block=False)
        )
        for n in range(downsample_blocks):
            self.model = SubmoduleWrapper(
                self.model, initial_channels*2**(downsample_blocks-n), 2,
                sa_block=sa_block, fixed_resample=fixed_resample, sn=sn,
                activation=activation, generator_block=generator_block)
        self.output_transform = generator_block(initial_channels, input_shape[0], sn=sn, down_block=False)
        self.apply(init_weights(nn.init.calculate_gain('relu')))
        
    def forward(self, x):
        x_i = self.input_transform(x)
        x_m = self.model(x_i)
        out = self.output_transform(x_m)
        out = torch.tanh(out)
        return out
    
class DiscriminatorFeatureExtractor(nn.Module):
    def __init__(self, input_shape, initial_channels=8, downsample_blocks=2, sa_block=False, fixed_resample=True, sn=spectral_norm, activation=lambda: nn.LeakyReLU(0.1), discriminator_block=DiscriminatorBlock):
        super().__init__()

        #self.input_transform = discriminator_block(input_shape[0], initial_channels, sn=sn, activation=activation)
        self.input_transform = sn(nn.Conv2d(input_shape[0], initial_channels, kernel_size=1))
        self.feature_extractor = nn.Sequential(
            discriminator_block(initial_channels, 2*initial_channels,
                               fixed_resample=fixed_resample, downsample=True, sn=sn, activation=activation),
            SelfAttentionBlock(2*initial_channels) if sa_block else nn.Identity(),
            *[discriminator_block(initial_channels*2**n, initial_channels*2**(n+1),
                                 fixed_resample=fixed_resample, downsample=True, sn=sn, activation=activation)
              for n in range(1, downsample_blocks)],
            discriminator_block(initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks, sn=sn, activation=activation)
        )
        #self.feature_compressor = nn.Sequential(
        #    sn(nn.Linear(initial_channels*2**downsample_blocks*(input_shape[1]//(2**downsample_blocks))**2, initial_channels*2**downsample_blocks)),
        #    activation(),
        #    sn(nn.Linear(initial_channels*2**downsample_blocks, initial_channels*2**downsample_blocks))
        #)
        
    def forward(self, x):
        x_i = self.input_transform(x)
        x_fe = self.feature_extractor(x_i)
        x_sp = x_fe.sum(dim=(2, 3)).view(-1, x_fe.shape[1])
        #x_sp = self.feature_compressor(x_fe.view(-1, np.prod(x_fe.shape[1:])))
        return x_sp

class DiscriminatorClassifier(nn.Module):
    def __init__(self, input_shape, n_logits=1, n_features=32, mixer=False, sn=spectral_norm, activation=lambda: nn.LeakyReLU(0.1)):
        super().__init__()
        
        if mixer:
            self.feature_mixer = sn(nn.Linear(n_features//2, n_features//2))
        else:
            self.feature_mixer = None
        self.classifier = sn(nn.Linear(n_features, n_logits))
        
    def mix_features(self, x):
        if self.feature_mixer is None:
            return x
        else:
            return self.feature_mixer(x)
    
    def classify_features(self, x):
        return self.classifier(x)

class SwaDiscriminatorClassifier(nn.Module):
    def __init__(self, discriminator_classifier, avg_fn):
        super().__init__()
        
        if discriminator_classifier.feature_mixer is not None:
            self.feature_mixer = torch.optim.swa_utils.AveragedModel(discriminator_classifier.feature_mixer, avg_fn=avg_fn)
        else:
            self.feature_mixer = None
        self.classifier = torch.optim.swa_utils.AveragedModel(discriminator_classifier.classifier, avg_fn=avg_fn)
    
    def mix_features(self, x):
        if self.feature_mixer is None:
            return x
        else:
            return self.feature_mixer(x)
    
    def classify_features(self, x):
        return self.classifier(x)
    
    def update_parameters(self, discriminator_classifier):
        if self.feature_mixer is not None:
            assert discriminator_classifier.feature_mixer is not None
            self.feature_mixer.update_parameters(discriminator_classifier.feature_mixer)
        self.classifier.update_parameters(discriminator_classifier.classifier)

class Classifier(nn.Module):
    def __init__(self, input_shape, leakage_classes=2, initial_channels=8, downsample_blocks=1, sa_block=True, fixed_resample=True):
        super().__init__()
        
        self.feature_extractor = DiscriminatorFeatureExtractor(
            input_shape, initial_channels=initial_channels, downsample_blocks=downsample_blocks, sa_block=sa_block,
            fixed_resample=fixed_resample, sn=lambda x: x, activation=nn.ReLU)
        self.leakage_classifier = DiscriminatorClassifier(
            input_shape, n_logits=leakage_classes, n_features=initial_channels*2**downsample_blocks,
            sn=lambda x: x, activation=nn.ReLU)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.leakage_classifier.classify_features(features)
        return out
        
class Discriminator(nn.Module):
    def __init__(self, input_shape, leakage_classes=2, initial_channels=8, downsample_blocks=2, sa_block=False, fixed_resample=False):
        super().__init__()
        
        discriminator_block = WideDiscriminatorBlock
        activation = lambda: nn.LeakyReLU(0.1)
        
        self.feature_extractor = DiscriminatorFeatureExtractor(
            input_shape, initial_channels=initial_channels, downsample_blocks=downsample_blocks, sa_block=sa_block,
            fixed_resample=fixed_resample, sn=spectral_norm, activation=activation, discriminator_block=discriminator_block)
        self.leakage_classifier = DiscriminatorClassifier(
            input_shape, n_logits=leakage_classes, n_features=initial_channels*2**downsample_blocks,
            sn=spectral_norm, activation=activation)
        self.realism_classifier = DiscriminatorClassifier(
            input_shape, n_logits=1, n_features=2*initial_channels*2**downsample_blocks,
            sn=spectral_norm, activation=activation)
        
    def extract_features(self, x):
        return self.feature_extractor(x)
    
    def mix_features_for_realism_analysis(self, x):
        return self.realism_classifier.mix_features(x)
    
    def classify_realism(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.realism_classifier.classify_features(x)
    
    def classify_leakage(self, x):
        return self.leakage_classifier.classify_features(x)
    
class SwaDiscriminator(nn.Module):
    def __init__(self, discriminator, avg_fn):
        super().__init__()
        
        self.feature_extractor = torch.optim.swa_utils.AveragedModel(discriminator.feature_extractor, avg_fn=avg_fn)
        self.leakage_classifier = SwaDiscriminatorClassifier(discriminator.leakage_classifier, avg_fn)
        self.realism_classifier = SwaDiscriminatorClassifier(discriminator.realism_classifier, avg_fn)
        
    def extract_features(self, x):
        return self.feature_extractor(x)
    
    def mix_features_for_realism_analysis(self, x):
        return self.realism_classifier.mix_features(x)
    
    def classify_realism(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return self.realism_classifier.classify_features(x)
    
    def classify_leakage(self, x):
        return self.leakage_classifier.classify_features(x)
    
    def update_parameters(self, discriminator):
        self.feature_extractor.update_parameters(discriminator.feature_extractor)
        self.leakage_classifier.update_parameters(discriminator.leakage_classifier)
        self.realism_classifier.update_parameters(discriminator.realism_classifier)