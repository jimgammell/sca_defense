import os
import numpy as np
from datasets.common import SavedNpzDataset

from utils import get_print_to_log
print = get_print_to_log(__file__)

class PurduePowerTraceDataset(SavedNpzDataset):
    def __init__(self,
                 trace_length,
                 byte,
                 trace_transform=None,
                 plaintext_transform=None,
                 ap_transform=None,
                 train=True,
                 data_path=None,
                 download_base = r'https://github.com/SparcLab/X-DeepSCA/raw/master/mat_traces',
                 download_urls=[r'cw308XGD2_10k_nov5_1447.zip',
                                r'cw308XGD3_10k_nov5_1643.zip',
                                r'cw308XGD4_10k_nov8_2228.zip',
                                r'cw308XGD5_10k_nov9_1538.zip',
                                r'cw308XGD6_10k_nov9_1559.zip',
                                r'cw308XGD7_10k_nov22_2022.zip',
                                r'cw308XGD8_50k_nov14_1635.zip',
                                r'cw308XGD9_nov14_2011.zip'],
                 test_indices = [0, 1, 2, 3, 4, 5],
                 data_partition_size=250):
        if data_path == None:
            d = os.path.join('.', 'saved_datasets')
        else:
            d = data_path
        if not os.path.isdir(d):
            os.mkdir(d)
        d = os.path.join(d, 'purdue_power_traces')
        if not os.path.isdir(d):
            import requests
            import zipfile
            import shutil
            from scipy import io
            print('Downloading Purdue power traces dataset...')
            try:
                print('Creating directory structure...')
                os.mkdir(d)
                os.mkdir(os.path.join(d, 'train'))
                os.mkdir(os.path.join(d, 'test'))
                temp_dir = os.path.join('.', 'temp')
                os.mkdir(temp_dir)
                mat_dir = os.path.join(temp_dir, 'mat')
                os.mkdir(mat_dir)
                print('Downloading and installing dataset...')
                shard_idx = 0
                for dl_idx, download_url in enumerate(download_urls):
                    compressed_filename = download_url.split('/')[-1]
                    print('Downloading file %s...'%(compressed_filename))
                    r = requests.get('/'.join((download_base, download_url)), allow_redirects=True, timeout=10)
                    with open(os.path.join(temp_dir, compressed_filename), 'wb') as F:
                        F.write(r.content)
                    print('Extracting file %s...'%(compressed_filename))
                    with zipfile.ZipFile(os.path.join(temp_dir, compressed_filename), 'r') as zip_ref:
                        zip_ref.extractall(mat_dir)
                    extracted_filename = '.'.join((compressed_filename.split('.')[0], 'mat'))
                    print('Parsing file %s...'%(extracted_filename))
                    data = io.loadmat(os.path.join(mat_dir, extracted_filename))
                    assert len(data['traces']) == len(data['textin']) == len(data['key'])
                    for idx in range(0, len(data['traces'])//data_partition_size):
                        if dl_idx in test_indices:
                            save_path = os.path.join(d, 'test', 'shard_%d.npz'%(shard_idx))
                        else:
                            save_path = os.path.join(d, 'train', 'shard_%d.npz'%(shard_idx))
                        with open(save_path, 'wb') as F:
                            np.savez(F,
                                     keys=data['key'][data_partition_size*idx:data_partition_size*(idx+1)],
                                     pts=data['textin'][data_partition_size*idx:data_partition_size*(idx+1)],
                                     traces=data['traces'][data_partition_size*idx:data_partition_size*(idx+1)])
                        shard_idx += 1
            except:
                shutil.rmtree(os.path.join('.', 'saved_datasets', 'purdue_power_traces'))
                assert False
            finally:
                shutil.rmtree(os.path.join('.', 'temp'))
        base_path = os.path.join(d, 'train' if train else 'test')
        super().__init__(base_path, 'key', trace_length, byte, trace_transform, plaintext_transform, ap_transform)