import torch
from torch import nn

def get_param_count(model):
    num_params = 0
    for _, parameter in model.named_parameters():
        num_params += parameter.numel()
    return num_params

class IdentityGenerator(nn.Module):
    def __init__(self, latent_dims, label_dims, output_shape, feature_dims=0, **kwargs):
        super().__init__()
        self.output_shape = output_shape
        self.feature_dims = feature_dims
        self.placeholder_transform = nn.Linear(1, 1, bias=False)

    def forward(self, latent_vars, labels, *args):
        return torch.zeros(labels.size(0), *self.output_shape[1:], device=latent_vars.get_device())
