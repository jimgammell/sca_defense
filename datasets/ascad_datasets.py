import os
import torch
from torch.utils.data import Dataset
from datasets.common import download_file, extract_zip
import numpy as np

class AscadDataset(Dataset):
    def __init__(self,
                 transform=None,
                 target_transform=None,
                 train=True,
                 download=True,
                 save_dir=os.path.join('.', 'saved_datasets', 'ascad'),
                 download_url=r'https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip'):
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
            import h5py
            temp_dir = os.path.join('.', 'saved_datasets', 'temp')
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            compressed_filename = 'ascad_dataset.zip'
            if not os.path.exists(os.path.join(temp_dir, compressed_filename)):
                r = requests.get(download_url, allow_redirects=True, timeout=10)
                with open(os.path.join(temp_dir, compressed_filename), 'wb') as F:
                    F.write(r.content)
            extracted_dir = os.path.join(temp_dir, 'extracted')
            if not os.path.exists(extracted_dir):
                os.mkdir(extracted_dir)
                with zipfile.ZipFile(os.path.join(temp_dir, compressed_filename), 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
            with h5py.File(os.path.join(extracted_dir, 'ASCAD_data', 'ASCAD_databases', 'ASCAD.h5'), 'r') as F:
                training_dataset = dict(labels=np.array(F['Profiling_traces']['labels']),
                                        traces=np.array(F['Profiling_traces']['traces']))
                testing_dataset = dict(labels=np.array(F['Attack_traces']['labels']),
                                       traces=np.array(F['Attack_traces']['traces']))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if not os.path.exists(os.path.join(save_dir, 'train')):
                os.mkdir(os.path.join(save_dir, 'train'))
            if not os.path.exists(os.path.join(save_dir, 'test')):
                os.mkdir(os.path.join(save_dir, 'test'))
            np.savez(os.path.join(save_dir, 'train', 'data.npz'), **training_dataset)
            np.savez(os.path.join(save_dir, 'test', 'data.npz'), **testing_dataset)
        
        if train:
            dataset = np.load(os.path.join(save_dir, 'train', 'data.npz'))
        else:
            dataset = np.load(os.path.join(save_dir, 'test', 'data.npz'))
        self.traces = dataset['traces'].astype(float)
        self.labels = dataset['labels']
        self.num_examples = len(self.traces)
        self.traces -= np.mean(self.traces)
        self.traces /= np.std(self.traces)**2
        assert self.num_examples == len(self.labels)
        self.transform = transform
        self.target_transform = target_transform
        self.trace_shape = self.traces[0].shape
        for trace in self.traces[1:]:
            assert trace.shape == self.trace_shape
        self.num_classes = len(np.unique(self.labels))
        
    def __getitem__(self, idx):
        trace = torch.from_numpy(self.traces[idx]).to(torch.float)
        label = torch.from_numpy(np.array(self.labels[idx])).to(torch.float)
        if self.transform != None:
            trace = self.transform(trace)
        if self.target_transform != None:
            label = self.target_transform(label)
        return trace, label
    
    def __len__(self):
        return self.num_examples