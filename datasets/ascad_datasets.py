import os
import torch
from torch import nn
from torchaudio.functional import lowpass_biquad, highpass_biquad
from torch.utils.data import Dataset
from datasets.common import download_file, extract_zip
import numpy as np

AES_Sbox = np.array([
            0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
            ])

class LowPassFilter:
    def __init__(self, cutoff_freq_range):
        self.filter = lowpass_biquad
        self.cutoff_freq_range = cutoff_freq_range
    def __call__(self, trace):
        cutoff_freq = np.random.uniform(*cutoff_freq_range)
        return self.filter(trace, 1e-3, cutoff_freq)

class HighPassFilter:
    def __init__(self, cutoff_freq_range):
        self.filter = highpass_biquad
        self.cutoff_freq_range = cutoff_freq_range
    def __call__(self, trace):
        cutoff_freq = np.random.uniform(*cutoff_freq_range)
        return self.filter(trace, 1e-3, cutoff_freq)

class RandomShift:
    def __init__(self, center_idx=650, max_shift_size=0):
        self.center_idx = center_idx
        self.max_shift_size = max_shift_size
    def __call__(self, trace):
        traces = []
        for channel in range(trace.shape[0]):
            shift = np.random.randint(low=-self.max_shift_size, high=self.max_shift_size+1)
            traces.append(trace[channel, self.center_idx-350+shift:self.center_idx+350+shift])
        traces = torch.cat(traces).view(trace.shape[0], -1)
        return traces

class RandomNoise:
    def __init__(self, max_convex_coef=0.05):
        self.max_convex_coef = max_convex_coef
    def __call__(self, trace):
        traces = []
        for channel in range(trace.shape[0]):
            convex_coef = np.random.uniform(self.max_convex_coef)
            noise = torch.randn_like(trace[channel, :])
            traces.append((1-convex_coef)*trace[channel, :] + convex_coef*noise)
        traces = torch.cat(traces).view(*trace.shape)
        return traces

class AscadDataset(Dataset):
    def __init__(self,
                 transform=None,
                 target_transform=None,
                 train=True,
                 byte=2,
                 traces_per_sample=1,
                 mixup=False,
                 download=True,
                 whiten_traces=True,
                 subtract_mean_trace=False,
                 save_dir=os.path.join('.', 'saved_datasets', 'ascad'),
                 download_url=r'https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip',
                 trace_padding=(300, 300),
                 target_points=(45400, 46100),
                 test_split_idx=50000):
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
            import h5py
            import shutil
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
            traces, plaintexts, ciphertexts, keys, masks = [], [], [], [], []
            with h5py.File(os.path.join(extracted_dir, 'ASCAD_data', 'ASCAD_databases', 'ATMega8515_raw_traces.h5'), 'r') as F:
                raw_traces = F['traces']
                raw_data = F['metadata']
                raw_plaintexts = raw_data['plaintext']
                raw_keys = raw_data['key']
                raw_masks = raw_data['masks']
                raw_traces_pois = raw_traces[:, target_points[0]-trace_padding[0]:target_points[1]+trace_padding[1]+1]
                raw_traces_profiling = raw_traces_pois[:test_split_idx]
                raw_traces_attack = raw_traces_pois[test_split_idx:]
                plaintexts_profiling = raw_plaintexts[:test_split_idx]
                plaintexts_attack = raw_plaintexts[test_split_idx:]
                keys_profiling = raw_keys[:test_split_idx]
                keys_attack = raw_keys[test_split_idx:]
                labels_profiling = np.uint8(AES_Sbox[plaintexts_profiling[:, byte]^keys_profiling[:, byte]])
                labels_attack = np.uint8(AES_Sbox[plaintexts_attack[:, byte]^keys_attack[:, byte]])
            
            training_dataset = dict(traces=np.array(raw_traces_profiling),
                                    labels=np.array(labels_profiling))
            testing_dataset = dict(traces=np.array(raw_traces_attack),
                                   labels=np.array(labels_attack))
            
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if not os.path.exists(os.path.join(save_dir, 'train')):
                os.mkdir(os.path.join(save_dir, 'train'))
            if not os.path.exists(os.path.join(save_dir, 'test')):
                os.mkdir(os.path.join(save_dir, 'test'))
            np.savez(os.path.join(save_dir, 'train', 'data.npz'), **training_dataset)
            np.savez(os.path.join(save_dir, 'test', 'data.npz'), **testing_dataset)
            shutil.rmtree(temp_dir)
        
        if train:
            dataset = np.load(os.path.join(save_dir, 'train', 'data.npz'))
        else:
            dataset = np.load(os.path.join(save_dir, 'test', 'data.npz'))
        
        traces = dataset['traces'].astype(float)
        labels = dataset['labels']
        if whiten_traces:
            mean = np.mean(traces[trace_padding[0]:-trace_padding[1]])
            traces -= mean
            std = np.std(traces[trace_padding[0]:-trace_padding[1]])
            traces /= std
        self.datapoints = dict()
        for label_val in np.unique(labels):
            self.datapoints[label_val] = traces[labels==label_val]
        
        self.num_examples = len(traces)
        assert self.num_examples == len(labels)
        self.transform = transform
        self.target_transform = target_transform
        trace_shape = traces[0].shape
        for trace in traces[1:]:
            assert trace.shape == trace_shape
        self.classes = np.array(list(self.datapoints.keys()))
        self.num_classes = len(self.classes)
        self.traces_per_sample = traces_per_sample
        
    def get_traces_for_label(self, label):
        return self.datapoints[label]
    
    def __getitem__(self, *_):
        label = np.random.choice(self.classes)
        trace_indices = np.random.choice(np.arange(len(self.datapoints[label])), size=self.traces_per_sample)
        traces = self.datapoints[label][trace_indices]
        label = torch.tensor(label).to(torch.long)
        traces = torch.from_numpy(traces).to(torch.float)
        if self.transform != None:
            traces = self.transform(traces)
        if self.target_transform != None:
            label = self.target_transform(label)
        return traces, label
    
    def __len__(self):
        return self.num_examples