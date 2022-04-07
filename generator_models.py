import numpy as np
import torch
from torch import nn

class CompositeGenerator(nn.Module):
    def __init__(self, generators):
        super().__init__()
        
        self.generators = generators
    
    def forward(self, x):
        (trace, plaintext, key) = x
        active_generator = self.generators[key]
        protected_trace = active_generator((trace, plaintext, key))
        return protected_trace

class Generator(nn.Module):
    def __init__(self, trace_map, plaintext_map, key_map, cumulative_map):
        super().__init__()
        
        self.trace_map = trace_map
        self.plaintext_map = plaintext_map
        self.key_map = key_map
        self.cumulative_map = cumulative_map
    
    def forward(self, x):
        (trace, plaintext, key) = x
        trace_image = self.trace_map(trace)
        plaintext_image = self.plaintext_map(plaintext)
        key_image = self.key_map(key)
        visible_trace = self.cumulative_map(torch.concat(trace_image,
                                                         plaintext_image,
                                                         key_image))
        return visible_trace

def get_zero_map(input_shape, output_shape):
    modules = [nn.Flatten(start_dim=1, end_dim=-1)]
    zero_layer = nn.Linear(input_features=np.prod(input_shape), output_features=np.prod(output_shape), bias=False)
    nn.init.zeros_(zero_layer.weights)
    zero_layer.requires_grad = False
    modules.append(zero_layer)
    modules.append(nn.Unflatten(dim=-1, unflattened_size=output_shape))
    model = nn.Sequential(*modules)
    return model
    
def get_identity_map(input_shape, output_shape):
    modules = [nn.Flatten(start_dim=1, end_dim=-1),
               nn.Identity(),
               nn.Unflatten(dim=-1, unflattened_size=output_shape)]
    model = nn.Sequential(*modules)
    return model

def get_mlp_map(input_shape, output_shape, layers, hidden_activation):
    modules = [nn.Flatten(start_dim=1, end_dim=-1)]
    layers.insert(0, np.prod(input_shape))
    modules.append(nn.Linear(input_features=np.prod(input_shape), output_features=layers[0]))
    for l1, l2 in zip(layers[:-1], layers[1:]):
        modules.append(hidden_activation())
        modules.append(nn.Linear(input_features=l1, output_features=l2))
    modules.append(nn.Unflatten(dim=-1, unflattened_size=output_shape))
    model = nn.Sequential(*modules)
    return model

def get_rnn_map(input_shape, output_shape, glimpse_length, layers, nonlinearity):
    class RnnMap(nn.Module):
        def __init__(self,
                     input_shape,
                     glimpse_length,
                     output_shape,
                     layers,
                     nonlinearity):
            super().__init__()
            
            self.input_reshape = nn.Flatten(start_dim=1, end_dim=-1)
            layers.insert(0, glimpse_length)
            self.recurrent_modules = []
            for (l1, l2) in zip(layers[:-1], layers[1:]):
                self.recurrent_modules.append(nn.RNNCell(input_size=l1, hidden_size=l2, nonlinearity=nonlinearity))
            self.recurrent_map = torch.Sequential(*recurrent_modules)
            output_modules = [nn.Linear(input_features=layers[-1], output_features=np.prod(output_shape)),
                              nn.Reshape(dim=-1, unflattened_size=output_shape)]
            self.output_map = torch.Sequential(*output_modules)
            self.glimpse_length = glimpse_length
        def forward(self, x):
            x = self.input_reshape(x)
            glimpses = torch.split(x, self.glimpse_length, dim=-1)
            glimpses[-1] = nn.functional.pad(glimpses[-1],
                                             (self.glimpse_length-glimpses[-1].size()[-1]))
            h = torch.randn_like(glimpses[0])
            for glimpse in glimpses:
                h = self.recurrent_map(glimpse, output)
            output = self.output_map(h)
            return output
        
    model = RnnMap(input_shape, glimpse_length, output_shape, layers, nonlinearity)
    return model
    