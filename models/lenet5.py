import torch
from torch import nn

class LeNet5(nn.Module):
    def __init__(self,
                 input_shape,
                 output_classes):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(input_shape[1],
                                             6,
                                             kernel_size=5,
                                             stride=1,
                                             padding=2,
                                             bias=True),
                                   nn.BatchNorm2d(6),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(6,
                                             16,
                                             kernel_size=5,
                                             stride=1,
                                             padding=0,
                                             bias=True),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Flatten())
        eg_input = torch.rand(input_shape)
        _ = self.model(eg_input)
        
        self.input_shape = input_shape
        self.output_classes = output_classes
        
    def forward(self, x):
        logits = self.model(x)
        return logits
    
    def __repr__(self):
        s = 'LeNet5 model:' +\
            '\n\tInput shape: {}'.format(self.input_shape) +\
            '\n\tOutput classes: {}'.format(self.output_classes) +\
            '\nModel summary:\n' + super(nn.Module, self).__init__()
        return s