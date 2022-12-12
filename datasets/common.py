import os
import requests
import zipfile

def download_dataset(zipped_dest, zipped_url=None, extract=True):
    if zipped_url is not None:
        os.makedirs(os.path.split(zipped_dest)[0], exist_ok=True)
        r = requests.get(zipped_url, allow_redirects=True, timeout=10)
        with open(zipped_dest, 'wb') as F:
            F.write(r.content)
    if extract:
        with zipfile.ZipFile(zipped_dest, 'r') as zip_ref:
            zip_ref.extractall(os.path.split(zipped_dest)[0])