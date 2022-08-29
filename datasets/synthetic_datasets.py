import numpy as np
from torch.utils.data import Dataset

from utils import get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))

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