import os
import requests
import zipfile

def download_dataset(zipped_dest, zipped_url=None, extracted_dest=None):
    if zipped_url is not None:
        os.makedirs(zipped_dest, exist_ok=True)
        r = requests.get(zipped_url, allow_redirects=True, timeout=10)
        with open(zipped_dest, 'wb') as F:
            F.write(r.content)
    if extracted_dest is not None:
        os.mkdirs(extracted_dest, exist_ok=True)
        with zipfile.ZipFile(zipped_dest, 'r') as zip_ref:
            zip_ref.extractall(extracted_dest)