import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from datasets.common import download_dataset

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

def ap_to_key_indices(plaintext):
    return AES_Sbox[plaintext ^ np.arange(256)]

def accumulate_predictions(p):
    return np.sum(np.log(p), axis=0)

class AscadDataset(Dataset):
    def __init__(self,
                 dataset_loc=os.path.join('.', 'raw_datasets', 'ascad'),
                 transform=None,
                 target_transform=None,
                 download=True,
                 train=True,
                 max_desync=0,
                 byte=2,
                 whiten_traces=True,
                 traces_per_sample=1):
        super().__init__()
        
        self.max_desync = max_desync
        self.byte = byte
        self.traces_per_sample = traces_per_sample
        self.transform = transform
        self.target_transform = target_transform
        
        # Download and extract the dataset, if required
        zipped_loc = os.path.join(dataset_loc, 'compressed_dataset.zip')
        extracted_loc = os.path.join(dataset_loc, 'ASCAD_data', 'ASCAD_databases', 'ATMega8515_raw_traces.h5')
        if os.path.exists(extracted_loc):
            pass
        elif os.path.exists(zipped_loc):
            download_dataset(zipped_loc, extract=True)
        elif download:
            download_dataset(zipped_loc,
                             zipped_url=r'https://www.data.gouv.fr/s/resources/ascad/20180530-163000/ASCAD_data.zip',
                             extract=True)
        else:
            raise Exception('Cannot load dataset without downloading because no extracted dataset was found at {} and no zipped dataset was found at {}.'.format(extracted_loc, zipped_loc))
            
        # Load the dataset from the extracted files
        in_file = h5py.File(extracted_loc, 'r')
        if train:
            start_idx, end_idx = 0, 50000
        else:
            start_idx, end_idx = 50000, 60000
        raw_traces = in_file['traces'][start_idx:end_idx]
        self.traces = raw_traces[:, 45400-max_desync:46100+max_desync].astype(float)
        if whiten_traces:
            self.traces -= np.mean(self.traces)
            self.traces /= np.std(self.traces)
        raw_data = in_file['metadata'][start_idx:end_idx]
        raw_plaintexts = raw_data['plaintext'][:, byte]
        raw_keys = raw_data['key'][:, byte]
        raw_masks = raw_data['masks']
        self.attack_points = np.uint8(AES_Sbox[raw_plaintexts ^ raw_keys])
        self.key = np.unique(raw_keys)[0]
        self.plaintexts = raw_plaintexts
        self.num_examples = len(self.attack_points)
        self.input_shape = (700, traces_per_sample)
        self.output_shape = (256, traces_per_sample)
    
    def __len__(self):
        return self.num_examples
        
    def get_traces_for_key(self, key):
        indices = self.key_locs[key]
        traces = self.traces[indices]
        if self.transform is not None:
            for idx, trace in enumerate(traces):
                traces[idx] = self.transform(trace)
        plaintexts = self.plaintexts[indices]
        return traces, plaintexts
    
    def reorder_logits(self, logits, plaintexts):
        for idx, (pt, lg) in enumerate(zip(logits, plaintexts)):
            logits[idx] = lg[ap_to_key_indices(pt)]
        return logits
        
    def __getitem__(self, idx):
        traces = [self.traces[idx]]
        attack_points = [self.attack_points[idx]]
        plaintexts = [self.plaintexts[idx]]
        for _ in range(self.traces_per_sample-1):
            idx = np.random.choice(self.key_locs[key])
            traces.append(self.traces[idx])
            attack_points.append(self.attack_points[idx])
            plaintexts.append(self.plaintexts[idx])
        if self.max_desync > 0:
            for idx, trace in enumerate(traces):
                desync = np.random.randint(2*self.max_desync)
                traces[idx] = trace[desync:desync+700]
        traces = np.stack(traces, axis=-1)
        attack_points = np.stack(attack_points, axis=-1)
        plaintexts = np.stack(plaintexts, axis=-1)
        if self.transform is not None:
            traces = self.transform(traces)
        if self.target_transform is not None:
            attack_points = self.target_transform(attack_points)
        return traces, attack_points, plaintexts