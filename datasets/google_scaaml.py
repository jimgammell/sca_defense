import os
import json
import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np
import requests
import zipfile
import shutil
from tqdm import tqdm

class GoogleScaamlDataset(Dataset):
    def __init__(self,
                 transform=None,
                 target_transform=None,
                 train=True,
                 download=True,
                 whiten_traces=True,
                 interval_to_use=[0, 20000],
                 downsample_ratio=4,
                 byte=0,
                 store_in_ram=False,
                 attack_point='sub_bytes_in',
                 save_dir=os.path.join('.', 'downloads', 'google_scaaml_dataset'),
                 download_url=r'https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip'):
        super().__init__()
        
        if download:
            current_settings = {
                'interval_to_use': interval_to_use,
                'downsample_ratio': downsample_ratio,
                'whiten_traces': whiten_traces
            }
            if os.path.exists(os.path.join(save_dir, 'existing_settings.json')):
                with open(os.path.join(save_dir, 'existing_settings.json'), 'r') as F:
                    existing_settings = json.load(F)
                up_to_date = current_settings == existing_settings
            else:
                up_to_date = False
            if not up_to_date:
                os.makedirs(os.path.join(save_dir, 'compressed'), exist_ok=True)
                os.makedirs(os.path.join(save_dir, 'extracted'), exist_ok=True)
                if not os.path.exists(os.path.join(save_dir, 'compressed', 'compressed_dataset.zip')):
                    print('Downloading GoogleScaamlDataset files.')
                    r = requests.get(download_url, allow_redirects=True, timeout=10)
                    with open(os.path.join(save_dir, 'compressed', 'compressed_dataset.zip'), 'wb') as F:
                        F.write(r.content)
                    print('\tDone.')
                with zipfile.ZipFile(os.path.join(save_dir, 'compressed', 'compressed_dataset.zip'), 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(save_dir, 'extracted'))
                for phase in ['train', 'test']:
                    for f in os.listdir(os.path.join(save_dir, 'extracted', 'datasets', 'tinyaes', phase)):
                        shard = dict(np.load(os.path.join(save_dir, 'extracted', 'datasets', 'tinyaes', phase, f)))
                        traces = torch.from_numpy(shard['traces']).to(torch.float)
                        if whiten_traces:
                            traces -= torch.mean(traces)
                            traces /= torch.std(traces)
                        traces = traces[:, interval_to_use[0]:interval_to_use[1]].transpose(-1, -2)
                        traces = nn.functional.max_pool1d(traces, kernel_size=downsample_ratio, stride=downsample_ratio)
                        shard['traces'] = traces.numpy()
                        np.savez(os.path.join(save_dir, 'extracted', 'datasets', 'tinyaes', phase, f), **shard)
                with open(os.path.join(save_dir, 'existing_settings.json'), 'w') as F:
                    json.dump(current_settings, F)
        
        self.phase = 'train' if train else 'test'
        self.shard_filenames = os.listdir(os.path.join(save_dir, 'extracted', 'datasets', 'tinyaes', self.phase))
        self.store_in_ram = store_in_ram
        self.save_dir = save_dir
        if store_in_ram:
            self.shards = {idx: np.load(os.path.join(
                save_dir, 'extracted', 'datasets', 'tinyaes', self.phase, f))
                           for idx, f in enumerate(self.shard_filenames)}
            
        self.num_shards = len(self.shard_filenames)
        self.samples_per_shard = len(self.get_shard(0)['traces'])
        self.transform = transform
        self.target_transform = target_transform
        self.byte = byte
        self.attack_point = attack_point
        self.interval_to_use = interval_to_use
        self.downsample_ratio = downsample_ratio
        self.whiten_traces = whiten_traces
        eg_trace, eg_label, _ = self.__getitem__(0)
        self.trace_shape = eg_trace.shape
        self.label_shape = eg_label.shape
        
    def get_shard(self, idx):
        if self.store_in_ram:
            return self.shards[idx]
        else:
            return np.load(os.path.join(
                self.save_dir, 'extracted', 'datasets', 'tinyaes', self.phase, self.shard_filenames[idx]))
        
    def __getitem__(self, idx):
        shard_idx = idx // self.samples_per_shard
        sample_idx = idx % self.samples_per_shard
        shard = self.get_shard(shard_idx)
        trace = shard['traces'][sample_idx]
        label = shard[self.attack_point][self.byte, sample_idx]
        trace = torch.from_numpy(trace).to(torch.float)
        label = torch.tensor(label).to(torch.long)
        metadata = {k: shard[k][self.byte, sample_idx] for k in shard.keys() if k not in ['traces', self.attack_point]}
        if self.transform is not None:
            trace = self.transform(trace)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return trace, label, metadata
    
    def __len__(self):
        return self.num_shards*self.samples_per_shard
    
    def __repr__(self):
        s = 'Google SCAAML TinyAES power trace dataset'
        s += '\n\tTraining phase: {}'.format(self.phase)
        s += '\n\tTrace shape: {}'.format(self.trace_shape)
        s += '\n\tLabel shape: {}'.format(self.label_shape)
        s += '\n\tNumber of shards: {}'.format(self.num_shards)
        s += '\n\tSamples per shard: {}'.format(self.samples_per_shard)
        s += '\n\tTransform: {}'.format(self.transform)
        s += '\n\tTarget transform: {}'.format(self.target_transform)
        s += '\n\tByte: {}'.format(self.byte)
        s += '\n\tAttack point: {}'.format(self.attack_point)
        s += '\n\tInterval to use: {}'.format(self.interval_to_use)
        s += '\n\tDownsampling ratio: {}'.format(self.downsample_ratio)
        s += '\n\tWhiten traces: {}'.format(self.whiten_traces)
        return s
