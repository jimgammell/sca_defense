import os
import torch
from torch.utils.data import Dataset
from datasets.common import download_file, extract_zip
import numpy as np

class PurdueDataset(Dataset):
    def __init__(self,
                 transform=None,
                 target_transform=None,
                 train=True,
                 download=True,
                 whiten_traces=True,
                 subtract_mean_trace=True,
                 save_dir=os.path.join('.', 'saved_datasets', 'purdue'),
                 download_urls=[os.path.join(r'https://github.com/SparcLab/X-DeepSCA/raw/master/mat_traces', zip_path)
                                for zip_path in [r'cw308XGD2_10k_nov5_1447.zip',
                                                 r'cw308XGD3_10k_nov5_1643.zip',
                                                 r'cw308XGD4_10k_nov8_2228.zip',
                                                 r'cw308XGD5_10k_nov9_1538.zip',
                                                 r'cw308XGD6_10k_nov9_1559.zip',
                                                 r'cw308XGD7_10k_nov22_2022.zip',
                                                 r'cw308XGD8_50k_nov14_1635.zip',
                                                 r'cw308XGD9_nov14_2011.zip']]):
        super().__init__()
        def save_dir_valid():
            return os.path.exists(save_dir)
        if download and not(save_dir_valid()):
            os.mkdir(save_dir)
            import requests
            import zipfile
            from scipy import io
            import shutil
            temp_dir = os.path.join('.', 'saved_datasets', 'temp')
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            extracted_dir = os.path.join(temp_dir, 'extracted')
            if not os.path.exists(extracted_dir):
                os.mkdir(extracted_dir)
            for download_url in download_urls:
                compressed_filename = download_url.split('/')[-1]
                if not os.path.exists(os.path.join(temp_dir, compressed_filename)):
                    r = requests.get(download_url, allow_redirects=True, timeout=10)
                    with open(os.path.join(temp_dir, compressed_filename), 'wb') as F:
                        F.write(r.content)
                with zipfile.ZipFile(os.path.join(temp_dir, compressed_filename), 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
                extracted_filename = '.'.join((compressed_filename.split('.')[0], 'mat'))
                data = io.loadmat(os.path.join(extracted_dir, extracted_filename))
                assert len(data['traces']) == len(data['textin']) == len(data['key'])
                save_name = '.'.join((compressed_filename.split('.')[0], 'npz'))
                with open(os.path.join(save_dir, save_name), 'wb') as F:
                    np.savez(F, keys=data['key'], pts=data['textin'], traces=data['traces'])
            shutil.rmtree(temp_dir)
        if train:
            device_paths = [r'cw308XGD2_10k_nov5_1447.npz',
                            r'cw308XGD3_10k_nov5_1643.npz',
                            r'cw308XGD4_10k_nov8_2228.npz',
                            r'cw308XGD5_10k_nov9_1538.npz']
        else:
            device_paths = [r'cw308XGD6_10k_nov9_1559.npz',
                            r'cw308XGD8_50k_nov14_1635.npz',
                            r'cw308XGD9_nov14_2011.npz']
        self.traces = []
        self.labels = []
        for path in device_paths:
            data = np.load(os.path.join(save_dir, path))
            self.traces.append(data['traces'][:, 0:500])
            self.labels.append(data['keys'][:, 0])
        self.traces = np.vstack(self.traces)
        self.labels = np.hstack(self.labels)
        self.num_examples = len(self.traces)
        
        def whiten_traces_fn():
            for idx, trace in enumerate(self.traces):
                self.traces[idx] -= np.mean(trace)
                self.traces[idx] /= np.std(trace)
        if whiten_traces:
            whiten_traces_fn()
        
        if subtract_mean_trace:
            mean_trace = np.zeros(self.traces[0].shape)
            for idx, trace in enumerate(self.traces):
                mean_trace = (idx/(idx+1))*mean_trace + (1/(idx+1))*trace
            for idx in range(len(self.traces)):
                self.traces[idx] -= mean_trace
            whiten_traces_fn()
        
        assert self.num_examples == len(self.labels)
        self.transform = transform
        self.target_transform = target_transform
        self.trace_shape = self.traces[0].shape
        for trace in self.traces[1:]:
            assert trace.shape == self.trace_shape
        self.num_classes = len(np.unique(self.labels))
    
    def __getitem__(self, idx):
        trace = torch.from_numpy(self.traces[idx]).to(torch.float)
        label = torch.from_numpy(np.array(self.labels[idx])).to(torch.long)
        if self.transform != None:
            trace = self.transform(trace)
        if self.target_transform != None:
            label = self.target_transform(label)
        return trace, label
    
    def __len__(self):
        return self.num_examples