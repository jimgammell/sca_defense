import os
import torch
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
                 save_dir = os.path.join('.', 'saved_datasets', 'google'),
                 download_url=r'https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip'):
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
            shutil.move(os.path.join(extracted_dir, 'datasets', 'tinyaes', 'train'),
                        os.path.join(save_dir, 'train'))
            shutil.move(os.path.join(extracted_dir, 'datasets', 'tinyaes', 'test'),
                        os.path.join(save_dir, 'test'))
            shutil.rmtree(temp_dir)