import os
from datasets.common import SavedNpzDataset

from utils import get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))

class GooglePowerTraceDataset(SavedNpzDataset):
    def __init__(self,
                 *args,
                 data_path=None,
                 train=True,
                 download_url=r'https://storage.googleapis.com/scaaml-public/scaaml_intro/datasets.zip',
                 **kwargs):
        if data_path == None:
            d = os.path.join('.', 'saved_datasets')
        else:
            d = data_path
        if not os.path.isdir(d):
            os.mkdir(d)
        d = os.path.join(d, 'google_scaaml')
        if not os.path.isdir(d):
            import requests
            import zipfile
            import shutil
            print('Downloading Google SCAAML dataset...')
            try:
                print('Creating directory structure...')
                os.mkdir(d)
                temp_dir = os.path.join('.', 'temp')
                os.mkdir(temp_dir)
                compressed_filename = 'google_scaaml_dataset.zip'
                print('Downloading zipped dataset...')
                r = requests.get(download_url, allow_redirects=True, timeout=10)
                with open(os.path.join(temp_dir, compressed_filename), 'wb') as F:
                    F.write(r.content)
                print('Extracting dataset...')
                extracted_dir = os.path.join(temp_dir, 'extracted')
                os.mkdir(extracted_dir)
                with zipfile.ZipFile(os.path.join(temp_dir, compressed_filename), 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
                print('Installing dataset...')
                shutil.move(os.path.join(extracted_dir, 'datasets', 'tinyaes', 'train'),
                            os.path.join(d, 'train'))
                shutil.move(os.path.join(extracted_dir, 'datasets', 'tinyaes', 'test'),
                            os.path.join(d, 'test'))
            except:
                shutil.rmtree(os.path.join('.', 'saved_datasets', 'google_scaaml'))
                assert False
            finally:
                shutil.rmtree(os.path.join('.', 'temp'))
        base_path = os.path.join(d, 'train' if train else 'test')
        super().__init__(base_path, *args, **kwargs)