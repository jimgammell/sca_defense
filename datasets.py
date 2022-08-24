import os
import requests
import zipfile
import shutil
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
                 num_keys,
                 trace_length,
                 byte,
                 plaintext_encoding,
                 train=True,
                 data_path=None,
                 download_url=r'https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip'):
        if data_path == None:
            d = os.path.join('.', 'datasets')
        else:
            d = data_path
        if not os.path.isdir(d):
            os.mkdir(d)
        d = os.path.join(d, 'google_scaaml')
        if not os.path.isdir(d):
            try:
                os.mkdir(d)
                d = os.path.join(d, 'train' if train else 'test')
                os.mkdir(d)
                temp_dir = os.path.join('.', 'temp')
                os.mkdir(temp_dir)
                compressed_filename = 'google_scaaml_dataset.zip'
                r = requests.get(download_url, allow_redirects=True, timeout=10)
                with open(os.path.join(temp_dir, compressed_filename), 'wb') as F:
                    F.write(r.content)
                extracted_dir = os.path.join(temp_dir, 'extracted')
                os.mkdir(extracted_dir)
                with zipfile.ZipFile(os.path.join(temp_dir, compressed_filename), 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
            except:
                shutil.rmtree(os.path.join('.', 'datasets', 'google_scaaml'))
                assert False
            finally:
                pass#shutil.rmtree(os.path.join('.', 'temp'))
        