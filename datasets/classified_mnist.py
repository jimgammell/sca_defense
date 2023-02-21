import numpy as np
import torch
import torchvision
from tqdm import tqdm

def add_square_watermark(image, center, radius):
    for ridx in range(center[0]-radius, center[0]+radius+1):
        for cidx in range(center[1]-radius, center[1]+radius+1):
            if (0 <= ridx < image.shape[0]) and (0 <= cidx < image.shape[1]) and ((np.abs(center[0]-ridx) == radius) or (np.abs(center[1]-cidx) == radius)):
                image[ridx, cidx] = np.max(image)
    return image

def add_plus_watermark(image, center, radius):
    for ridx in range(center[0]-radius, center[0]+radius+1):
        for cidx in range(center[1]-radius, center[1]+radius+1):
            if (0 <= ridx < image.shape[0]) and (0 <= cidx < image.shape[1]) and ((ridx == center[0]) or (cidx == center[1])):
                image[ridx, cidx] = np.max(image)
    return image

class WatermarkedMNIST(torchvision.datasets.MNIST):
    def __init__(self, *args, deterministic_position=False, deterministic_radius=False, normalize=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.orig_data = len(self.data)*[np.zeros(self.data[0].shape, dtype=np.float)]
        self.watermarked_data = len(self.data)*[np.zeros(self.data[0].shape, dtype=np.float)]
        self.watermark_targets = len(self.targets)*[0]
        for idx, data in enumerate(self.data):
            data = np.array(data).astype(np.float)
            data = 2.*(data/np.max(data))-1.
            self.orig_data[idx] = data
            watermark_target = np.random.randint(2)
            if deterministic_position:
                watermark_center = (24, 24)
            else:
                watermark_center = np.random.randint(2, 26, size=(2,))
            if deterministic_radius:
                watermark_radius = 3
            else:
                watermark_radius = np.random.randint(1, 6)
            if watermark_target == 1:
                watermarked_data = add_square_watermark(data, watermark_center, watermark_radius)
            else:
                watermarked_data = add_plus_watermark(data, watermark_center, watermark_radius)
            self.watermarked_data[idx] = watermarked_data
            self.watermark_targets[idx] = watermark_target
        
    def __getitem__(self, idx):
        _, target = super().__getitem__(idx)
        orig_data = self.orig_data[idx]
        orig_data = torch.from_numpy(np.array(orig_data)).to(torch.float).unsqueeze(0)
        data = torch.from_numpy(self.watermarked_data[idx]).to(torch.float).unsqueeze(0)
        if self.transform is not None:
            data = self.transform(data)
        watermark_target = torch.tensor(self.watermark_targets[idx]).to(torch.long)
        return data, watermark_target, {'orig_image': orig_data, 'target': target}