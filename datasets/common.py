import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.datasets import MNIST

import datasets
from utils import get_print_to_log, list_module_attributes, get_filename
print = get_print_to_log(get_filename(__file__))

def preprocess_transform(transform_constructor, transform_kwargs):
    if transform_constructor == None:
        return None
    if type(transform_constructor) == list:
        assert type(transform_kwargs) == list
        assert len(transform_constructor) == len(transform_kwargs)
    else:
        transform_constructor = [transform_constructor]
        transform_kwargs = [transform_kwargs]
    transforms = []
    for idx, (constructor, kwargs) in enumerate(zip(transform_constructor, transform_kwargs)):
        if type(constructor) == str:
            if constructor in list_module_attributes(datasets.transforms):
                constructor = getattr(datasets.transforms, constructor)
            elif constructor in list_module_attributes(torchvision.transforms):
                constructor = getattr(torchvision.transforms, constructor)
            else:
                assert False
        transform = constructor(**kwargs)
        transforms.append(transform)
    full_transform = torchvision.transforms.Compose(transforms)
    return full_transform

class SavedNpzDataset(Dataset):
    def __init__(self,
                 base_path,
                 attack_point,
                 trace_length,
                 byte,
                 trace_transform=None,
                 trace_transform_kwargs={},
                 plaintext_transform=None,
                 plaintext_transform_kwargs={},
                 ap_transform=None,
                 ap_transform_kwargs={},
                 data_storage_device=None):
        super().__init__()
        
        self.files = [f for f in os.listdir(base_path) if f.split('.')[-1] == 'npz']
        self.data_storage_device = data_storage_device
        def get_shard_dict(shard):
            shard_rv = {'traces': shard['traces'][:, :trace_length]}
            if (shard['pts'].shape[0] == 16) and (shard[attack_point].shape[0] == 16):
                shard_rv.update({'pts': shard['pts'][byte, :],
                                 attack_point: shard[attack_point][byte, :]})
            else:
                shard_rv.update({'pts': shard['pts'][:, byte],
                                 attack_point: shard[attack_point][:, byte]})
            return shard_rv
        if self.data_storage_device == None:
            self.get_shard = lambda filename: get_shard_dict(np.load(os.path.join(base_path, filename)))
        else:
            assert (self.data_storage_device == 'ram') or ('cuda' in self.data_storage_device)
            self.dataset_on_device = {}
            for filename in self.files:
                shard = np.load(os.path.join(base_path, filename))
                shard_to_store = get_shard_dict(shard)
                for key in shard_to_store.keys():
                    shard_to_store[key] = torch.tensor(shard_to_store[key]).to('cpu' if data_storage_device == 'ram'
                                                                               else data_storage_device)
                self.dataset_on_device[filename] = shard_to_store
            self.get_shard = lambda filename: self.dataset_on_device[filename]
        eg_shard = self.get_shard(self.files[0])
        self.shard_size = len(eg_shard['traces'])
        self.num_examples = self.shard_size*len(self.files)
        self.base_path = base_path
        self.attack_point = attack_point
        self.trace_length = trace_length
        self.byte = byte
        self.trace_transform = preprocess_transform(trace_transform, trace_transform_kwargs)
        self.plaintext_transform = preprocess_transform(plaintext_transform, plaintext_transform_kwargs)
        self.ap_transform = preprocess_transform(ap_transform, ap_transform_kwargs)
    
    def __getitem__(self, idx):
        shard_idx = idx // self.shard_size
        trace_idx = idx % self.shard_size
        shard = self.get_shard(self.files[shard_idx])
        trace = shard['traces'][trace_idx]
        plaintext = shard['pts'][trace_idx]
        ap = shard[self.attack_point][trace_idx]
        if self.trace_transform != None:
            trace = self.trace_transform(trace)
        if self.plaintext_transform != None:
            plaintext = self.plaintext_transform(plaintext)
        if self.ap_transform != None:
            ap = self.ap_transform(ap)
        return trace, ap #trace, plaintext, ap
    
    def __len__(self):
        return self.num_examples
    
    def __repr__(self):
        eg_trace, eg_pt, eg_ap = self.__getitem__(0)
        s = 'Dataset from saved npz files:' +\
            '\n\tBase path: %s'%(self.base_path) +\
            '\n\tData storage device: {}'.format(self.data_storage_device) +\
            '\n\tTotal examples: %d'%(self.__len__()) +\
            '\n\tExamples per shard: %d'%(self.shard_size) +\
            '\n\tTarget attack point: %s'%(self.attack_point) +\
            '\n\tTarget byte: %d'%(self.byte) +\
            '\n\tTrace length: %d'%(self.trace_length) +\
            '\n\tTrace transform: {}'.format(self.trace_transform) +\
            '\n\tPlaintext transform: {}'.format(self.plaintext_transform) +\
            '\n\tAttack point transform: {}'.format(self.ap_transform) +\
            '\n\tTrace shape: {}'.format(eg_trace.shape) +\
            '\n\tPlaintext shape: {}'.format(eg_pt.shape) +\
            '\n\tAttack point shape: {}'.format(eg_ap.shape)
        return s