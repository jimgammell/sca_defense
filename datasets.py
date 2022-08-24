import os
import requests
import zipfile
import shutil
import random
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

class GooglePowerTraceDataset(Dataset):
    def __init__(self,
                 attack_point,
                 trace_length,
                 byte,
                 plaintext_encoding,
                 num_keys=None,
                 train=True,
                 data_path=None,
                 download_url=r'https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip'):
        super().__init__()
        if data_path == None:
            d = os.path.join('.', 'datasets')
        else:
            d = data_path
        if not os.path.isdir(d):
            os.mkdir(d)
        d = os.path.join(d, 'google_scaaml')
        if not os.path.isdir(d):
            print('Downloading Google SCAAML dataset...')
            try:
                print('Creating directory structure...')
                os.mkdir(d)
                temp_dir = os.path.join('.', 'temp')
                os.mkdir(temp_dir)
                compressed_filename = 'google_scaaml_dataset.zip'
                print('Downloading zipped dataset...')
                r = requests.get(download_url, allow_redirects=True, timeout=10)
                with open(os.path.join(temp_dir, compressed_filename), 'wb') as F:
                    F.write(r.content)
                print('Extracting dataset...')
                extracted_dir = os.path.join(temp_dir, 'extracted')
                os.mkdir(extracted_dir)
                with zipfile.ZipFile(os.path.join(temp_dir, compressed_filename), 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
                print('Installing dataset...')
                shutil.move(os.path.join(extracted_dir, 'datasets', 'tinyaes', 'train'),
                            os.path.join(d, 'train'))
                shutil.move(os.path.join(extracted_dir, 'datasets', 'tinyaes', 'test'),
                            os.path.join(d, 'test'))
            except:
                shutil.rmtree(os.path.join('.', 'datasets', 'google_scaaml'))
                assert False
            finally:
                shutil.rmtree(os.path.join('.', 'temp'))
        self.attack_point = attack_point
        self.num_keys = num_keys
        self.trace_length = trace_length
        self.byte = byte
        base_path = os.path.join(d, 'train' if train else 'test')
        self.get_shard = lambda filename: np.load(os.path.join(base_path, filename))
        self.files = [f for f in os.listdir(base_path) if f.split('.')[-1] == 'npz']
        if num_keys != None:
            self.files = random.choices(self.files, k=num_keys)
        self.num_examples = 0
        for file in self.files:
            shard = self.get_shard(file)
            self.num_examples += len(shard['traces'])
        self.shard_size = len(shard['traces'])
        
    def __getitem__(self, idx):
        shard_idx = idx // self.shard_size
        trace_idx = idx % self.shard_size
        shard = self.get_shard(self.files[shard_idx])
        trace = shard['traces'][trace_idx]
        plaintext = shard['pts'][self.byte]
        ap = shard[self.attack_point][self.byte]
        return trace, plaintext, ap
    
    def __len__(self):
        return self.num_examples
        
        