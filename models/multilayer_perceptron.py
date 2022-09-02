import numpy as np
from torch import nn

from utils import get_print_to_log, get_filename
from models.common import get_param_count
print = get_print_to_log(get_filename(__file__))

class MultilayerPerceptron(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs=256,
                 hidden_layers=[],
                 hidden_activation=nn.ReLU,
                 batch_norm=False,
                 batch_norm_kwargs={},
                 dropout=0.):
        super().__init__()
        layer_sizes = [n_inputs] + hidden_layers + [n_outputs]
        if type(hidden_activation) == list:
            assert len(hidden_activation) == len(hidden_layers)
        else:
            hidden_activation = len(hidden_layers)*[hidden_activation]
        if batch_norm == False:
            batch_norm = (len(layer_sizes)-1)*[False]
        elif batch_norm == True:
            batch_norm = (len(layer_sizes)-1)*[True]
        else:
            assert len(batch_norm) == len(layer_sizes)-1
        if type(dropout) == float:
            dropout = (len(layer_sizes)-1)*[dropout]
        else:
            assert len(dropout) == len(layer_sizes)-1
        modules = []
        if batch_norm[0] != False:
            modules.append(nn.BatchNorm1d(layer_sizes[0], **batch_norm_kwargs))
        if dropout[0] != 0.:
            modules.append(nn.Dropout(dropout[0]))
        for idx in range(len(layer_sizes)-2):
            modules.append(nn.Linear(layer_sizes[idx], layer_sizes[idx+1]))
            if batch_norm[idx+1] != False:
                modules.append(nn.BatchNorm1d(layer_sizes[idx+1], **batch_norm_kwargs))
            modules.append(hidden_activation[idx]())
            if dropout[idx+1] != 0.:
                modules.append(nn.Dropout(dropout[idx+1]))
        modules.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.model = nn.Sequential(*modules)
        
        self.layer_sizes = layer_sizes
        self.hidden_activation = hidden_activation
        self.batch_norm = batch_norm
        self.batch_norm_kwargs = batch_norm_kwargs
        self.dropout = dropout
        
    def forward(self, x):
        return self.model(x)

class Linear(MultilayerPerceptron):
    def __init__(self, eg_input_shape):
        n_inputs = np.prod(eg_input_shape[1:])
        super().__init__(n_inputs)
        self.input_transform = nn.Flatten(1, -1)
        
    def forward(self, x):
        transformed_x = self.input_transform(x)
        return super().forward(transformed_x)
        
    def __repr__(self):
        s = 'Linear model:' +\
            '\n\tLayer sizes: {}'.format(self.layer_sizes) +\
            '\n\tParameter count: {}'.format(get_param_count(self)) +\
            '\nModel summary:\n{}'.format(self.model)
        return s

class XDeepSca(MultilayerPerceptron):
    def __init__(self,
                 eg_input_shape,
                 hidden_layers=[200, 200],
                 hidden_activation=nn.ReLU,
                 batch_norm=[False, True, True],
                 dropout=[0., .1, .05]):
        n_inputs = np.prod(eg_input_shape[1:])
        super().__init__(n_inputs,
                         hidden_layers=hidden_layers,
                         hidden_activation=hidden_activation,
                         batch_norm=batch_norm,
                         dropout=dropout)
        self.input_transform = nn.Flatten(1, -1)
        
    def forward(self, x):
        transformed_x = self.input_transform(x)
        return super().forward(transformed_x)
    
    def __repr__(self):
        s = 'X-DeepSCA model:' +\
            '\n\tLayer sizes: {}'.format(self.layer_sizes) +\
            '\n\tHidden activations: {}'.format(self.hidden_activation) +\
            '\n\tBatch norm: {}'.format(self.batch_norm) +\
            '\n\tDropout: {}'.format(self.dropout) +\
            '\n\tParameter count: {}'.format(get_param_count(self)) +\
            '\nModel summary:\n{}'.format(self.model)
        return s

class StandardGanGenerator(MultilayerPerceptron):
    def __init__(self,
                 latent_dims,
                 image_shape,
                 conditions = [],
                 output_transform=nn.Tanh,
                 hidden_layers=[128, 256, 512, 1024],
                 hidden_activation=lambda: nn.LeakyReLU(0.2),
                 batch_norm=[False, False, True, True, True],
                 batch_norm_kwargs={'momentum': 0.2}):
        
        self.latent_dims = latent_dims
        self.image_shape = image_shape
        if len(conditions) > 0:
            self.conditional_gan = True
            self.conditions = conditions
            n_inputs = latent_dims+len(conditions)
        else:
            self.conditional_gan = False
            n_inputs = latent_dims
            
        super().__init__(n_inputs=n_inputs,
                         n_outputs=int(np.prod(image_shape[1:])),
                         hidden_layers=hidden_layers,
                         hidden_activation=hidden_activation,
                         batch_norm=batch_norm,
                         batch_norm_kwargs=batch_norm_kwargs)
        
        if self.conditional_gan:
            self.condition_embedding = nn.Embedding(len(conditions), len(conditions))
        self.input_transform = nn.Flatten(1, -1)
        if type(output_transform) == str:
            output_transform = getattr(nn, output_transform)
        self.output_transform = output_transform()
    
    def forward(self, *args):
        if self.conditional_gan:
            (x, labels) = args
            transformed_x = self.input_transform(x)
            embedded_labels = self.condition_embedding(labels)
            logits = super().forward(torch.cat((transformed_x, embedded_labels), dim=1))
        else:
            (x,) = args
            transformed_x = self.input_transform(x)
            logits = super().forward(transformed_x)
        output = self.output_transform(logits).view(-1, *self.image_shape[1:])
        return output
    
    def __repr__(self):
        s = 'Standard GAN Generator model.' +\
            '\n\tLatent dimensions: {}'.format(self.latent_dims) +\
            '\n\tGenerated image shape: {}'.format(self.image_shape) +\
            '\n\tOutput activation: {}'.format(self.output_transform) +\
            '\n\tLayer sizes: {}'.format(self.layer_sizes) +\
            '\n\tHidden activations: {}'.format(self.hidden_activation) +\
            '\n\tBatch norm: {}'.format(self.batch_norm) +\
            '\n\tBatch norm kwargs: {}'.format(self.batch_norm_kwargs) +\
            '\n\tDropout: {}'.format(self.dropout) +\
            '\n\tParameter count: {}'.format(get_param_count(self)) +\
            '\nModel summary:\n{}'.format(self.model)
        return s

class StandardGanDiscriminator(MultilayerPerceptron):
    def __init__(self,
                 image_shape,
                 conditions=[],
                 n_outputs=1,
                 output_transform=nn.Sigmoid,
                 hidden_layers=[512, 256, 1],
                 hidden_activation=lambda: nn.LeakyReLU(0.2)):
        
        self.image_shape = image_shape
        if len(conditions) > 0:
            self.conditional_gan = True
            self.conditions = conditions
            n_inputs = np.prod(image_shape[1:]) + len(conditions)
        else:
            self.conditional_gan = False
            n_inputs = np.prod(image_shape[1:])
            
        super().__init__(n_inputs=n_inputs,
                         n_outputs=n_outputs,
                         hidden_layers=hidden_layers,
                         hidden_activation=hidden_activation,)
        
        if self.conditional_gan:
            self.condition_embedding = nn.Embedding(len(conditions), len(conditions))
        self.input_transform = nn.Flatten(1, -1)
        if type(output_transform) == str:
            output_transform = getattr(nn, output_transform)
        self.output_transform = output_transform()
    
    def forward(self, *args):
        if self.conditional_gan:
            (x, labels) = args
            transformed_x = self.input_transform(x)
            embedded_labels = self.condition_embedding(labels)
            logits = super().forward(torch.cat((transformed_x, embedded_labels), dim=1))
        else:
            (x,) = args
            transformed_x = self.input_transform(x)
            logits = super().forward(transformed_x)
        output = self.output_transform(logits)
        return output
    
    def __repr__(self):
        s = 'Standard GAN Discriminator model.' +\
            '\n\tConditional GAN: {}'.format(
            'True with conditions {}'.format(self.conditions)
            if self.conditional_gan else 'False') +\
            '\n\tInput image shape: {}'.format(self.image_shape) +\
            '\n\tOutput activation: {}'.format(self.output_transform) +\
            '\n\tLayer sizes: {}'.format(self.layer_sizes) +\
            '\n\tHidden activations: {}'.format(self.hidden_activation) +\
            '\n\tBatch norm: {}'.format(self.batch_norm) +\
            '\n\tBatch norm kwargs: {}'.format(self.batch_norm_kwargs) +\
            '\n\tDropout: {}'.format(self.dropout) +\
            '\n\tParameter count: {}'.format(get_param_count(self)) +\
            '\nModel summary:\n{}'.format(self.model)
        return s