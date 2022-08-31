from copy import copy
import numpy as np
import torch
from torch import distributions

from utils import get_print_to_log, get_filename, get_attribute_from_package
print = get_print_to_log(get_filename(__file__))

class RandomAffineTransform:
    def __init__(self,
                 m_distr_name, m_distr_kwargs,
                 b_distr_name, b_distr_kwargs):
        m_distr_constructor = get_attribute_from_package(m_distr_name, distributions)
        b_distr_constructor = get_attribute_from_package(b_distr_name, distributions)
        self.m_distr = m_distr_constructor(**m_distr_kwargs)
        self.b_distr = b_distr_constructor(**b_distr_kwargs)
    def __call__(self, x):
        m = self.m_distr.sample()
        b = self.b_distr.sample()
        transformed_x = m*x+b
        return transformed_x
    def __repr__(self):
        return '%s(m~%s, b~%s)'%(
            self.__class__.__name__,
            self.m_distr.__repr__(),
            self.b_distr.__repr__())

class IndependentElementNoise:
    def __init__(self,
                 noise_distr_name, noise_distr_kwargs,
                 trace_shape):
        noise_distr_constructor = get_attribute_from_package(noise_distr_name, distributions)
        for key, kwarg in copy(noise_distr_kwargs).items():
            if '__TRACE_SHAPE' in key:
                del noise_distr_kwargs[key]
                key = key.split('__TRACE_SHAPE')[0]
                kwarg = kwarg*torch.ones(trace_shape)
                noise_distr_kwargs.update({key: kwarg})
        self.noise_distr = noise_distr_constructor(**noise_distr_kwargs)
    def __repr__(self):
        return '%s(noise~%s)'%(
            self.__class__.__name__,
            self.noise_distr.__repr__())
    
class AdditiveRandomNoise(IndependentElementNoise):
    def __call__(self, x):
        noise = self.noise_distr.sample()
        transformed_x = x+noise
        return transformed_x

class MultiplicativeRandomNoise(IndependentElementNoise):
    def __call__(self, x):
        noise = self.noise_distr.sample()
        transformed_x = x*noise
        return transformed_x

class IntToOnehot:
    def __init__(self, classes):
        self.classes = classes
    def __call__(self, x):
        x_oh = np.zeros((self.classes,), dtype=int)
        x_oh[x] = 1
        return x_oh
    def __repr__(self):
        return self.__class__.__name__ + '()'

class IntToBinary:
    def __init__(self, bits):
        self.bits = bits
    def __call__(self, x):
        x_bin = np.zeros((self.bits,), dtype=int)
        pwr = self.bits
        while x > 0:
            pwr -= 1
            if x >= 1<<pwr:
                x -= 1<<pwr
                x_bin[self.bits-pwr-1] = 1
        return x_bin
    def __repr__(self):
        return self.__class__.__name__ + '()'

class IntToInt:
    def __init__(self):
        pass
    def __call__(self, x):
        return torch.tensor(x).type(torch.long).squeeze()
    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToTensor1D:
    def __init__(self):
        pass
    def __call__(self, x):
        return torch.tensor(x).type(torch.float).squeeze().unsqueeze(0)
    def __repr__(self):
        return self.__class__.__name__ + '()'