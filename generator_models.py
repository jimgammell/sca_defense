import numpy as np
import torch
from torch import nn
from utils import log_print as print
from copy import copy
from dataset import NormTensorMagnitude

class KeyOnlyGenerator(nn.Module):
    def __init__(self, key_trace_map):
        super().__init__()
        self.key_trace_map = key_trace_map
        self.output_transform = NormTensorMagnitude(1, -1)
    
    def forward(self, *args):
        _, trace, _, key = args
        protective_trace = self.key_trace_map(key)
        visible_trace = trace + protective_trace
        visible_trace = self.output_transform(visible_trace)
        return visible_trace

class CompositeGenerator(nn.Module):
    def __init__(self, generators):
        super().__init__()
        
        self.generators = nn.ModuleList()
        for (idx, key) in enumerate(generators.keys()):
            self.generators.append(generators[key])
        self.output_transform = NormTensorMagnitude(1, -1)
    
    def forward(self, *args):
        key_idx, trace, plaintext, key = args
        visible_traces = []
        for (idx, tr, pt, ky) in zip(torch.unbind(key_idx), torch.unbind(trace), torch.unbind(plaintext), torch.unbind(key)):
            tr = tr.unsqueeze(0)
            pt = pt.unsqueeze(0)
            ky = ky.unsqueeze(0)
            active_generator = self.generators[idx]
            protected_trace = active_generator((tr, pt, ky))
            visible_trace = tr + protected_trace
            visible_trace = self.output_transform(visible_trace)
            visible_traces.append(visible_trace)
        visible_traces = torch.stack(visible_traces)
        return visible_traces

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
        cumulative_input = torch.cat((trace_image, plaintext_image, key_image), dim=1)
        visible_trace = self.cumulative_map(cumulative_input)
        return visible_trace

def get_zero_map(input_shape, output_shape):
    modules = [nn.Flatten(start_dim=1, end_dim=-1)]
    zero_layer = nn.Linear(in_features=np.prod(input_shape), out_features=np.prod(output_shape), bias=False)
    nn.init.zeros_(zero_layer.weight)
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
    layers = copy(layers)
    modules = [nn.Flatten(start_dim=1, end_dim=-1)]
    layers.append(np.prod(output_shape))
    modules.append(nn.Linear(in_features=np.prod(input_shape), out_features=layers[0]))
    for l1, l2 in zip(layers[:-1], layers[1:]):
        modules.append(hidden_activation())
        modules.append(nn.Linear(in_features=l1, out_features=l2))
    modules.append(nn.Unflatten(dim=-1, unflattened_size=output_shape))
    model = nn.Sequential(*modules)
    return model

def get_rnn_map(input_shape, output_shape, glimpse_length, layers, nonlinearity):
    layers = copy(layers)
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
            output_modules = [nn.Linear(in_features=layers[-1], out_features=np.prod(output_shape)),
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
    