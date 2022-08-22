import numpy as np
import torch
from torch.utils.data import Dataset

class ToTensor:
    def __init__(self):
        pass
    
    def __call__(self, x):
        return torch.tensor(x).to(torch.float)

class BinaryDataset(Dataset):
    def __init__(self,
                 x_transform=None,
                 y_transform=None):
        super().__init__()
        self.x_transform = x_transform
        self.y_transform = y_transform
        
    def __getitem__(self, idx):
        bit_generator = lambda idx: np.array(idx % 2).astype(np.float)
        x = bit_generator(idx)
        y = bit_generator(idx)
        if self.x_transform != None:
            x = self.x_transform(x)
        if self.y_transform != None:
            y = self.y_transform(y)
        return x, y
    
    def __len__(self):
        return 2