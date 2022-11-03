import os
import shutil
import requests
import zipfile

def validate_dest(dest_dir, dest_name):
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    if os.path.exists(os.path.join(dest_dir, dest_name)):
        shutil.move(os.path.join(dest_dir, dest_name),
                    os.path.join(dest_dir, '.'.join((dest_name, 'BACKUP'))))

def download_file(src_url, dest_dir, dest_name):
    validate_dest(dest_dir, dest_name)
    r = requests.get(src_url, allow_redirects=True, timeout=10)
    with open(os.path.join(dest_dir, dest_name), 'w') as F:
        F.write(r.content)

def extract_zip(src_path, dest_dir, dest_name):
    validate_dest(dest_dir, dest_name)
    with zipfile.ZipFile(src_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(dest_dir, dest_name))