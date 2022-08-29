import numpy as np
import torch

from utils import get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))

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
        return torch.tensor(x).type(torch.float).squeeze()
    def __repr__(self):
        return self.__class__.__name__ + '()'