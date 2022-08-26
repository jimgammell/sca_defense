import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import get_print_to_log
print = get_print_to_log(__file__)

class SavedNpzDataset(Dataset):
    def __init__(self,
                 base_path,
                 attack_point,
                 trace_length,
                 byte,
                 trace_transform=None,
                 plaintext_transform=None,
                 ap_transform=None):
        super().__init__()
        
        self.get_shard = lambda filename: np.load(os.path.join(base_path, filename))
        self.files = [f for f in os.listdir(base_path) if f.split('.')[-1] == 'npz']
        eg_shard = self.get_shard(self.files[0])
        self.shard_size = len(eg_shard['traces'])
        self.num_examples = self.shard_size*len(self.files)
        self.base_path = base_path
        self.attack_point = attack_point
        self.trace_length = trace_length
        self.byte = byte
        self.trace_transform = trace_transform
        self.plaintext_transform = plaintext_transform
        self.ap_transform = ap_transform
    
    def __getitem__(self, idx):
        shard_idx = idx // self.shard_size
        trace_idx = idx % self.shard_size
        shard = self.get_shard(self.files[shard_idx])
        trace = shard['traces'][trace_idx][:self.trace_length]
        plaintext = shard['pts'][self.byte]
        ap = shard[self.attack_point][self.byte]
        if self.trace_transform != None:
            trace = self.trace_transform(trace)
        if self.plaintext_transform != None:
            plaintext = self.plaintext_transform(plaintext)
        if self.ap_transform != None:
            ap = self.ap_transform(ap)
        return trace, plaintext, ap
    
    def __len__(self):
        return self.num_examples
    
    def __repr__(self):
        s = 'Dataset from saved npz files:' +\
            '\n\tBase path: %s'%(self.base_path) +\
            '\n\tTotal examples: %d'%(self.__len__()) +\
            '\n\tExamples per shard: %d'%(self.shard_size) +\
            '\n\tTarget attack point: %s'%(self.attack_point) +\
            '\n\tTarget byte: %d'%(self.byte) +\
            '\n\tTrace length: %d'%(self.trace_length) +\
            '\n\tTrace transform: {}'.format(self.trace_transform) +\
            '\n\tPlaintext transform: {}'.format(self.plaintext_transform) +\
            '\n\tAttack point transform: {}'.format(self.ap_transform)
        return s