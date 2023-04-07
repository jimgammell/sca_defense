import os
import shutil
import json
from datasets.common import *
import gdown
from zipfile import ZipFile
import tarfile
import os
from collections import OrderedDict
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
import torch
from torch.utils.data import Dataset

def download_file(url, dest):
    if not os.path.exists(dest):
        gdown.download(url, dest, quiet=False)

def download_and_extract_dataset(url, dest, remove=True):
    download_file(url, dest)
    if dest.endswith('.tar.gz'):
        tar = tarfile.open(dest, 'r:gz')
        tar.extractall(os.path.dirname(dest))
        tar.close()
    elif dest.endswith('.tar'):
        tar = tarfile.open(dest, 'r:')
        tar.extractall(os.path.dirname(dest))
        tar.close()
    elif dest.endswith('.zip'):
        zf = ZipFile(dest, 'r')
        zf.extractall(os.path.dirname(dest))
        zf.close()
    if remove:
        os.remove(dest)

def get_default_transform():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat(3*[x]) if x.size(0)==1 else x),
        transforms.Lambda(lambda x: 2*(x-0.5*(x.max()-x.min()))/(x.max()-x.min()))
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def get_label_transform():
    return transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.long))

def get_augmentation_transform():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat(3*[x]) if x.size(0)==1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

class MultiDomainDataset(Dataset):
    def __init__(self, root, domains_to_use, url=None, download_extension=None, download=True, train=False, data_transform=get_default_transform(), target_transform=get_label_transform()):
        super().__init__()
        
        self.domains_to_use = domains_to_use
        self.data_transform = data_transform
        self.target_transform = target_transform
        
        assert all(domain in os.listdir(root) for domain in domains_to_use)
        self.environments = [
            DomainDataset(domain, root, data_transform=data_transform, target_transform=target_transform)
            for domain in self.domains_to_use
        ]
        self.num_datapoints = sum(len(d) for d in self.environments)
        self.classes = list(self.environments[0].data_files.keys())
        
    def __getitem__(self, idx):
        for d_idx, d in enumerate(self.environments):
            if idx < d.__len__():
                x, y = d.__getitem__(idx)
                env_idx = d_idx
                break
            else:
                idx -= d.__len__()
        return x, env_idx, {'target': y}
    
    def __len__(self):
        return self.num_datapoints
    
    def get_domain_name(self, idx):
        return self.domains_to_use[idx]
    
    def get_class_name(self, idx):
        return list(self.environments[0].data_files.keys())[idx]
        
class DomainDataset(Dataset):
    def __init__(self, domain, root=None, data_transform=None, target_transform=None):
        super().__init__()
        
        self.dataset_path = os.path.join(root, domain)
        assert os.path.exists(self.dataset_path)
        self.data_files = OrderedDict()
        for class_name in sorted(os.listdir(self.dataset_path)):
            self.data_files[class_name] = [f for f in os.listdir(os.path.join(self.dataset_path, class_name))]
        self.total_datapoints = sum(len(item) for item in self.data_files.values())
        self.to_pil_image = ToPILImage()
        self.data_transform = data_transform
        self.target_transform = target_transform
        
    def __getitem__(self, idx):
        for y, (class_name, class_files) in enumerate(self.data_files.items()):
            if idx < len(class_files):
                x = read_image(os.path.join(self.dataset_path, class_name, class_files[idx]))
                x = self.to_pil_image(x)
                break
            else:
                idx -= len(class_files)
        if self.data_transform is not None:
            x = self.data_transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
    
    def __len__(self):
        return self.total_datapoints

class OfficeHome(MultiDomainDataset):
    domains = ['Art', 'Clipart', 'Product', 'Real World']
    num_leakage_classes = 4
    num_downstream_classes = 65
    input_shape = (3, 64, 64)
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        assert all(domain in OfficeHome.domains for domain in domains_to_use)
        root = os.path.join('.', 'downloads', 'OfficeHomeDataset')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            download_and_extract_dataset(r'https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC',
                                         os.path.join('.', 'downloads', 'temp.zip'))
            os.rename(os.path.join('.', 'downloads', 'OfficeHomeDataset_10072016'), root)
        del kwargs['root']
        super().__init__(root, domains_to_use, **kwargs)

class VLCS(MultiDomainDataset):
    domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    num_classes = 5
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        assert all(domain in VLCS.domains for domain in domains_to_use)
        root = os.path.join('.', 'downloads', 'VLCS')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            download_and_extract_dataset(r'https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8',
                                         os.path.join('.', 'downloads', 'temp.tar.gz'))
        super().__init__(root, domains_to_use, **kwargs)

class PACS(MultiDomainDataset):
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    num_classes = 7
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        root = os.path.join('.', 'downloads', 'PACS')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            download_and_extract_dataset(r'https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd',
                                         os.path.join('.', 'downloads', 'temp.zip'))
            os.rename(os.path.join('.', 'downloads', 'kfold'), root)
        super().__init__(root, domains_to_use, **kwargs)

class Sviro(MultiDomainDataset):
    domains = ['aclass', 'escape', 'hilux', 'i3', 'lexus', 'tesla', 'tiguan', 'tucson', 'x5', 'zoe']
    num_classes = 7
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        root = os.path.join('.', 'downloads', 'Sviro')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            download_and_extract_dataset(r'https://sviro.kl.dfki.de/?wpdmdl=1731',
                                 os.path.join('.', 'downloads', 'temp.zip'))
            os.rename(os.path.join('.', 'downloads', 'SVIRO_DOMAINBED'), root)
        super().__init__(root, domains_to_use, **kwargs)

class DomainNet(MultiDomainDataset):
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    num_classes = 345
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains        
        root = os.path.join('.', 'downloads', 'DomainNet')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            for idx, url in enumerate([
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
                r'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip'
            ]):
                download_and_extract_dataset(url, os.path.join('.', 'downloads', 'DomainNet', 'temp_{}.zip'.format(idx)))
            download_file(r'https://github.com/facebookresearch/DomainBed/raw/main/domainbed/misc/domain_net_duplicates.txt',
                          os.path.join('.', 'downloads', 'DomainNet', 'duplicates.txt'))
            with open(os.path.join('.', 'downloads', 'DomainNet', 'duplicates.txt'), 'r') as F:
                for line in F.readlines():
                    try:
                        os.remove(os.path.join(root, line.strip()))
                    except OSError:
                        pass
        super().__init__(root, domains_to_use, **kwargs)

class TerraIncognita(MultiDomainDataset):
    domains = ['location_100', 'location_38', 'location_43', 'location_46']
    num_classes = 10
    
    def __init__(self, domains_to_use='all', download=True, **kwargs):
        if domains_to_use == 'all':
            domains_to_use = self.__class__.domains
        root = os.path.join('.', 'downloads', 'TerraIncognita')
        if not os.path.exists(root):
            assert download
            os.makedirs(root, exist_ok=True)
            for idx, url in enumerate([
                r'https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz',
                r'https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip'
            ]):
                download_and_extract_dataset(
                    url, os.path.join('.', 'downloads', 'TerraIncognita', 'temp_{}'.format(idx)+'.tar.gz' if idx==0 else '.zip')
                )
        if all(f in os.listdir(root) for f in ['eccv_18_all_images_sm', 'caltech_images_20210113.json']):
            include_locations = ['38', '46', '100', '43']
            include_categories = [
                'bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon', 'squirrel'
            ]
            images_folder = os.path.join(root, 'eccv_18_all_images_sm')
            annotations_file = os.path.join(root, 'caltech_images_20210113.json')
            destination_folder = root
            stats = {}
            with open(annotations_file, 'r') as F:
                data = json.load(F)
            category_dict = {}
            for item in data['categories']:
                category_dict[item['id']] = item['name']
            for image in data['images']:
                image_location = image['location']
                if image_location not in include_locations:
                    continue
                loc_folder = os.path.join(destination_folder, 'location_{}'.format(image_location))
                os.makedirs(loc_folder, exist_ok=True)
                image_id = image['id']
                image_fname = image['file_name']
                for annotation in data['annotations']:
                    if annotation['image_id'] == image_id:
                        if image_location not in stats.keys():
                            stats[image_location] = {}
                        category = category_dict[annotation['category_id']]
                        if category not in include_categories:
                            continue
                        if category not in stats[image_location]:
                            stats[image_location][category] = 0
                        else:
                            stats[image_location][category] += 1
                        loc_cat_folder = os.path.join(loc_folder, category)
                        os.makedirs(loc_cat_folder, exist_ok=True)
                        dst_path = os.path.join(loc_cat_folder, image_fname)
                        src_path = os.path.join(images_folder, image_fname)
                        if os.path.exists(src_path):
                            shutil.move(src_path, dst_path)
            shutil.rmtree(images_folder)
            os.remove(annotations_file)
        super().__init__(root, domains_to_use, **kwargs)