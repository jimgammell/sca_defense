import os
import json
import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np
import gdown
import zipfile
import shutil
from tqdm import tqdm

class GoogleScaamlDataset(Dataset):
    def __init__(self,
                 transform=None,
                 train=True,
                 download=True,
                 whiten_traces=True,
                 interval_to_use=[0, 20000],
                 downsample_ratio=4,
                 bytes=[0],
                 store_in_ram=False,
                 attack_points=['sub_bytes_in'],
                 save_dir=os.path.join('.', 'downloads', 'google_scaaml_dataset'),
                 download_url=r'https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip'):
        super().__init__()
        
        self.phase = 'train' if train else 'test'
        if download:
            current_settings = {
                'interval_to_use': interval_to_use,
                'downsample_ratio': downsample_ratio,
                'whiten_traces': whiten_traces
            }
            if os.path.exists(os.path.join(save_dir, 'existing_settings_{}.json'.format(self.phase))):
                with open(os.path.join(save_dir, 'existing_settings_{}.json'.format(self.phase)), 'r') as F:
                    existing_settings = json.load(F)
                up_to_date = current_settings == existing_settings
            else:
                up_to_date = False
            if not up_to_date:
                os.makedirs(os.path.join(save_dir, 'compressed'), exist_ok=True)
                os.makedirs(os.path.join(save_dir, 'extracted'), exist_ok=True)
                if not os.path.exists(os.path.join(save_dir, 'compressed', 'compressed_dataset.zip')):
                    print('Downloading GoogleScaamlDataset files.')
                    gdown.download(download_url, os.path.join(save_dir, 'compressed', 'compressed_dataset.zip'), quiet=False)
                    print('\tDone.')
                with zipfile.ZipFile(os.path.join(save_dir, 'compressed', 'compressed_dataset.zip'), 'r') as zip_ref:
                    os.makedirs(os.path.join(save_dir, 'extracted_tmp'), exist_ok=True)
                    zip_ref.extractall(os.path.join(save_dir, 'extracted_tmp'))
                    if os.path.exists(os.path.join(save_dir, 'extracted', 'datasets', 'tinyaes', self.phase)):
                        shutil.rmtree(os.path.join(save_dir, 'extracted', 'datasets', 'tinyaes', self.phase))
                    shutil.copytree(
                        os.path.join(save_dir, 'extracted_tmp', 'datasets', 'tinyaes', self.phase),
                        os.path.join(save_dir, 'extracted', 'datasets', 'tinyaes', self.phase)
                    )
                    shutil.rmtree(os.path.join(save_dir, 'extracted_tmp'))
                print('Preprocessing traces.')
                for f in tqdm(os.listdir(os.path.join(save_dir, 'extracted', 'datasets', 'tinyaes', self.phase))):
                    shard = dict(np.load(os.path.join(save_dir, 'extracted', 'datasets', 'tinyaes', self.phase, f)))
                    traces = torch.from_numpy(shard['traces']).to(torch.float)
                    if whiten_traces:
                        traces /= 0.5*(traces.max() - traces.min())
                        traces -= 0.5*(traces.max() + traces.min())
                    traces = traces[:, interval_to_use[0]:interval_to_use[1]].permute(0, 2, 1)
                    traces = nn.functional.max_pool1d(traces, kernel_size=downsample_ratio, stride=downsample_ratio)
                    shard['traces'] = traces.numpy()
                    np.savez(os.path.join(save_dir, 'extracted', 'datasets', 'tinyaes', self.phase, f), **shard)
                with open(os.path.join(save_dir, 'existing_settings_{}.json'.format(self.phase)), 'w') as F:
                    json.dump(current_settings, F)
        
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
        self.bytes = bytes
        self.attack_points = attack_points
        self.interval_to_use = interval_to_use
        self.downsample_ratio = downsample_ratio
        self.whiten_traces = whiten_traces
        eg_trace, eg_label = self.__getitem__(0)
        self.trace_shape = eg_trace.shape
        self.label_shapes = {key: item.shape for key, item in eg_label.items()}
        
    def get_shard(self, idx):
        if self.store_in_ram:
            return self.shards[idx]
        else:
            return np.load(os.path.join(
                self.save_dir, 'extracted', 'datasets', 'tinyaes', self.phase, self.shard_filenames[idx]))
        
    def __getitem__(self, idx, use_transform=True):
        shard_idx = idx // self.samples_per_shard
        sample_idx = idx % self.samples_per_shard
        shard = self.get_shard(shard_idx)
        trace = shard['traces'][sample_idx]
        labels = {
            '{}__{}'.format(tap, tb): torch.tensor(shard[tap][tb, sample_idx], dtype=torch.long)
            for tap in self.attack_points for tb in self.bytes
        }
        if use_transform and (self.transform is not None):
            trace = self.transform(trace)
        return trace, labels
    
    def __len__(self):
        return self.num_shards*self.samples_per_shard
    
    def __repr__(self):
        s = 'Google SCAAML TinyAES power trace dataset'
        s += '\n\tTraining phase: {}'.format(self.phase)
        s += '\n\tTrace shape: {}'.format(self.trace_shape)
        s += '\n\tLabel shapes: {}'.format(self.label_shapes)
        s += '\n\tNumber of shards: {}'.format(self.num_shards)
        s += '\n\tSamples per shard: {}'.format(self.samples_per_shard)
        s += '\n\tTransform: {}'.format(self.transform)
        s += '\n\tBytes: {}'.format(self.bytes)
        s += '\n\tAttack points: {}'.format(self.attack_points)
        s += '\n\tInterval to use: {}'.format(self.interval_to_use)
        s += '\n\tDownsampling ratio: {}'.format(self.downsample_ratio)
        s += '\n\tWhiten traces: {}'.format(self.whiten_traces)
        return s
