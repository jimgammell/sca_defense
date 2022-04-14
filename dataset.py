import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import log_print as print
from copy import copy

class NormTensorMagnitude:
    def __init__(self, mx, mn):
        self.min = mn
        self.max = mx
    def __call__(self, x):
        x = x - .5*(torch.max(x)+torch.min(x))
        x = x / torch.max(x)
        x = x * .5*(self.max-self.min)
        x = x + .5*(self.max+self.min)
        return x
    def __repr__(self):
        return self.__class__.__name__ + '()'

class IntToOnehot:
    def __init__(self, classes):
        self.classes = classes
    def __call__(self, x):
        x_oh = np.zeros((self.classes), dtype=int)
        x_oh[x] = 1
        return x_oh
    def __repr__(self):
        return self.__class__.__name__ + '()'

class IntToBinary:
    def __init__(self, bits):
        self.bits = bits
    def __call__(self, x):
        x = copy(x)
        x_bin = np.zeros((self.bits), dtype=int)
        pwr = self.bits
        while x > 0:
            pwr -= 1
            assert pwr >= 0
            if x >= 2**pwr:
                x -= 2**pwr
                x_bin[self.bits-pwr-1] = 1
        return x_bin
    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToTensor1D:
    def __init__(self):
        pass
    def __call__(self, x):
        return torch.tensor(x).type(torch.float).squeeze().unsqueeze(0)
    def __repr__(self):
        return self.__class__.__name__ + '()'

def download_dataset(dest):
    import requests
    
    urls = [r'https://github.com/SparcLab/X-DeepSCA/raw/master/mat_traces/cw308XGD2_10k_nov5_1447.zip',
            r'https://github.com/SparcLab/X-DeepSCA/raw/master/mat_traces/cw308XGD3_10k_nov5_1643.zip',
            r'https://github.com/SparcLab/X-DeepSCA/raw/master/mat_traces/cw308XGD4_10k_nov8_2228.zip',
            r'https://github.com/SparcLab/X-DeepSCA/raw/master/mat_traces/cw308XGD5_10k_nov9_1538.zip',
            r'https://github.com/SparcLab/X-DeepSCA/raw/master/mat_traces/cw308XGD6_10k_nov9_1559.zip',
            r'https://github.com/SparcLab/X-DeepSCA/raw/master/mat_traces/cw308XGD7_10k_nov22_2022.zip',
            r'https://github.com/SparcLab/X-DeepSCA/raw/master/mat_traces/cw308XGD8_50k_nov14_1635.zip',
            r'https://github.com/SparcLab/X-DeepSCA/raw/master/mat_traces/cw308XGD9_nov14_2011.zip']
    
    compressed_dir = os.path.join(dest, 'compressed_data')
    if not(os.path.isdir(compressed_dir)):
        os.mkdir(compressed_dir)
        
    for url in urls:
        url_name = url.split('/')[-1]
        if os.path.exists(os.path.join(compressed_dir, url_name)):
            continue
        try:
            r = requests.get(url, allow_redirects=True, timeout=10)
            with open(os.path.join(compressed_dir, url_name), 'wb') as F:
                F.write(r.content)
        except:
            print('Failed to download file from {}.'.format(url))

def extract_dataset(dest, delete_download_afterwards=False):
    import zipfile
    
    extracted_dir = os.path.join(dest, 'extracted_data')
    if not(os.path.isdir(extracted_dir)):
        os.mkdir(extracted_dir)
    
    compressed_dir = os.path.join(dest, 'compressed_data')
    zip_files = [f for f in os.listdir(compressed_dir) if '.zip' in f]
    for f in zip_files:
        with zipfile.ZipFile(os.path.join(compressed_dir, f), 'r') as zip_ref:
            if all([os.path.exists(os.path.join(extracted_dir, ff)) for ff in zip_ref.namelist()]):
                pass
            else:
                zip_ref.extractall(extracted_dir)
        if delete_download_afterwards:
            os.remove(os.path.join(compressed_dir, f))

def preprocess_dataset(dest, delete_extracted_after_preprocess=False):
    from scipy import io
    
    processed_dir = os.path.join(dest, 'processed_data')
    if not(os.path.isdir(processed_dir)):
        os.mkdir(processed_dir)
    
    extracted_dir = os.path.join(dest, 'extracted_data')
    extracted_files = [os.path.join(extracted_dir, f) for f in os.listdir(extracted_dir)  if f[-4:] == '.mat']
    plaintexts, traces, keys = [], [], []
    for f in extracted_files:
        data = io.loadmat(f)
        plaintext = data['textin']
        trace = data['traces']
        key = data['key']
        plaintexts.append(plaintext)
        traces.append(trace)
        keys.append(key)
    plaintexts = np.concatenate(plaintexts)
    traces = np.concatenate(traces)
    keys = np.concatenate(keys)
    
    num_bytes = len(keys[0])
    for pt in plaintexts:
        assert len(pt) == num_bytes
    for key in keys:
        assert len(key) == num_bytes
    
    for byte in range(num_bytes):
        for key in range(256):
            processed_filename = 'byte_%x__key_%02x.pickle'%(byte, key)
            if os.path.exists(os.path.join(processed_dir, processed_filename)):
                continue
            matching_ptxt, matching_tr = [], []
            for (pt, k, t) in zip(plaintexts, keys, traces):
                if k[byte] == key:
                    matching_ptxt.append(pt[byte])
                    matching_tr.append(t)
            matching_ptxt = np.array(matching_ptxt)
            traces = np.array(traces)
            with open(os.path.join(processed_dir, processed_filename), 'wb') as F:
                pickle.dump((matching_ptxt, matching_tr), F)
                
    if delete_extracted_after_preprocess:
        for f in extracted_files:
            os.remove(os.path.join(extracted_dir, f))

class AesKeyGroupDataset(Dataset):
    def idx_to_key_idx_pair(self, idx):
        for (kd_idx, key_dataset) in enumerate(self.key_datasets):
            if idx >= len(key_dataset):
                idx -= len(key_dataset)
            else:
                break
        return (kd_idx, idx)
    
    def __init__(self,
                 key_datasets,
                 byte=0,
                 key_transform=None):
        super().__init__()
        
        self.key_datasets = key_datasets
        self.key_transform = key_transform
        self.byte = byte
        self.num_samples = np.sum([len(kd) for kd in key_datasets])
        eg_key_idx, eg_trace, eg_plaintext, eg_key = self.__getitem__(0)
        self.key_idx_size = eg_key_idx.shape
        self.plaintext_size = eg_plaintext.size()
        self.trace_size = eg_trace.size()
        self.key_size = eg_key.size()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        (kd_idx, smp_idx) = self.idx_to_key_idx_pair(idx)
        key_dataset = self.key_datasets[kd_idx]
        key_idx = np.array(key_dataset.key, dtype=int)
        if self.key_transform != None:
            key = self.key_transform(key_idx)
        plaintext, trace = key_dataset.__getitem__(smp_idx)
        
        return key_idx, trace, plaintext, key
    
    def __repr__(self):
        s = ''
        s += self.__class__.__name__ + ':' + '\n'
        s += '\tAvailable keys: {}'.format([k.key for k in self.key_datasets]) + '\n'
        s += '\tKey transform: {}'.format(self.key_transform) + '\n'
        s += '\tByte: {}'.format(self.byte) + '\n'
        s += '\tNumber of samples available: {}'.format(self.num_samples) + '\n'
        s += '\tTrace size: {}'.format(self.trace_size) + '\n'
        s += '\tKey size: {}'.format(self.key_size) + '\n'
        s += '\tPlaintext size: {}'.format(self.plaintext_size) + '\n'
        s += '\tKey index size: {}'.format(self.key_idx_size)
        return s
            
class AesSingleKeyDataset(Dataset):
    def getitem_from_memory(self, idx):
        plaintext = self.plaintexts[idx]
        trace = self.traces[idx]
        return plaintext, trace
    
    def getitem_from_disk(self, idx):
        with open(os.path.join(self.data_filepath, self.data_filename), 'rb') as F:
            plaintexts, traces = pickle.load(F)
        plaintext = plaintexts[idx]
        trace = traces[idx]
        return plaintext, trace
    
    def __init__(self,
                 byte=0,
                 key=0,
                 trace_transform=None,
                 plaintext_transform=None,
                 keep_data_in_memory=True,
                 data_path=r'./data',
                 download=True,
                 extract=True,
                 preprocess=True,
                 delete_download_after_extraction=False,
                 delete_extracted_after_preprocess=False):
        super().__init__()
        
        self.key = key
        self.byte = byte
        self.trace_transform = trace_transform
        self.plaintext_transform = plaintext_transform
        self.keep_data_in_memory = keep_data_in_memory
        self.data_path = data_path
        self.data_filename = 'byte_%x__key_%02x.pickle'%(byte, key)
        
        if download:
            download_dataset(self.data_path)
        if extract:
            extract_dataset(self.data_path, delete_download_afterwards=delete_download_after_extraction)
        if preprocess:
            preprocess_dataset(self.data_path, delete_extracted_after_preprocess=delete_extracted_after_preprocess)
        
        self.data_path = os.path.join(self.data_path, 'processed_data')
        with open(os.path.join(self.data_path, self.data_filename), 'rb') as F:
            plaintexts, traces = pickle.load(F)
        assert len(plaintexts) == len(traces)
        self.num_examples = len(plaintexts)
                
        if self.keep_data_in_memory:
            self.getitem_fn = self.getitem_from_memory
            self.plaintexts = plaintexts
            self.traces = traces
        else:
            self.getitem_fn = self.getitem_from_disk
        
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        plaintext, trace = self.getitem_fn(idx)
        if self.plaintext_transform:
            plaintext = self.plaintext_transform(plaintext)
        if self.trace_transform:
            trace = self.trace_transform(trace)
        return plaintext, trace