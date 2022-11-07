import os
import torch
from torch import nn
from torch.utils.data import Dataset
from datasets.common import download_file, extract_zip
import numpy as np

class GoogleDataset(Dataset):
    def __init__(self,
                 transform=None,
                 target_transform=None,
                 train=True,
                 download=True,
                 whiten_traces=True,
                 subtract_mean_trace=True,
                 trace_length=20000,
                 byte=0,
                 attack_point='sub_bytes_in',
                 save_dir = os.path.join('.', 'saved_datasets', 'google'),
                 download_url=r'https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip'):
        assert whiten_traces == False
        assert subtract_mean_trace == False
        super().__init__()
        def save_dir_valid():
            if not os.path.exists(save_dir):
                return False
            elif not os.path.exists(os.path.join(save_dir, 'train')):
                return False
            elif not os.path.exists(os.path.join(save_dir, 'test')):
                return False
            else:
                return True
        if download and not(save_dir_valid()):
            import requests
            import zipfile
            import shutil
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            temp_dir = os.path.join('.', 'saved_datasets', 'temp')
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            compressed_filename = 'google_dataset.zip'
            if not os.path.exists(os.path.join(temp_dir, compressed_filename)):
                r = requests.get(download_url, allow_redirects=True, timeout=10)
                with open(os.path.join(temp_dir, compressed_filename), 'wb') as F:
                    F.write(r.content)
            extracted_dir = os.path.join(temp_dir, 'extracted')
            if not os.path.exists(extracted_dir):
                os.mkdir(extracted_dir)
                with zipfile.ZipFile(os.path.join(temp_dir, compressed_filename), 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
            train_data = {'traces': [], 'labels': []}
            for f in os.listdir(os.path.join(extracted_dir, 'datasets', 'tinyaes', 'train')):
                shard = np.load(os.path.join(extracted_dir, 'datasets', 'tinyaes', 'train', f))
                traces = torch.from_numpy(shard['traces'][:, :trace_length]).to(torch.float).transpose(-1, -2)
                traces = nn.functional.max_pool1d(traces, kernel_size=4, stride=4)
                train_data['traces'].append(traces.numpy())
                labels = shard[attack_point][byte, :]
                train_data['labels'].append(labels)
            train_data['traces'] = np.concatenate(train_data['traces'])
            train_data['labels'] = np.concatenate(train_data['labels'])
            if not os.path.exists(os.path.join(save_dir, 'train')):
                os.mkdir(os.path.join(save_dir, 'train'))
            np.savez(os.path.join(save_dir, 'train', 'data.npz'), **train_data)
            test_data = {'traces': [], 'labels': []}
            for f in os.listdir(os.path.join(extracted_dir, 'datasets', 'tinyaes', 'test')):
                shard = np.load(os.path.join(extracted_dir, 'datasets', 'tinyaes', 'test', f))
                traces = torch.from_numpy(shard['traces'][:, :trace_length]).to(torch.float).transpose(-1, -2)
                traces = nn.functional.max_pool1d(traces, kernel_size=4, stride=4)
                test_data['traces'].append(traces.numpy())
                labels = shard[attack_point][byte, :]
                test_data['labels'].append(labels)
            test_data['traces'] = np.concatenate(test_data['traces'])
            test_data['labels'] = np.concatenate(test_data['labels'])
            if not os.path.exists(os.path.join(save_dir, 'test')):
                os.mkdir(os.path.join(save_dir, 'test'))
            np.savez(os.path.join(save_dir, 'test', 'data.npz'), **test_data)
        if train:
            data = np.load(os.path.join(save_dir, 'train', 'data.npz'))
        else:
            data = np.load(os.path.join(save_dir, 'test', 'data.npz'))
        self.traces = data['traces']
        self.labels = data['labels']
        self.num_examples = len(self.traces)
        assert self.num_examples == len(self.labels)
        self.trace_length = trace_length
        self.byte = byte
        self.attack_point = attack_point
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        trace = torch.from_numpy(self.traces[idx]).to(torch.float)
        label = torch.tensor(self.labels[idx]).to(torch.long)
        if self.transform != None:
            trace = self.transform(trace)
        if self.target_transform != None:
            label = self.target_transform(label)
        return trace, label
    
    def __len__(self):
        return self.num_examples