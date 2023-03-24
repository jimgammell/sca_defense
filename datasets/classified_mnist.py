import numpy as np
import torch
import torchvision
from tqdm import tqdm
from gan_train import to_uint8

def normalize_tensor(x):
    r_x = np.max(x)-np.min(x)
    x = 2*(x-0.5*r_x)/r_x
    return x

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

@torch.no_grad()
def apply_transform(data, transform, batch_size, device):
    for b_idx in range(len(data)//batch_size):
        batch = data[batch_size*b_idx : batch_size*(b_idx+1)]
        batch = torch.stack([
            torch.from_numpy(x).to(device).to(torch.float) for x in batch
        ]).reshape((batch_size, -1, 28, 28))
        transformed_batch = transform(batch)
        transformed_batch = to_uint8(transformed_batch)
        for x_idx in range(batch_size):
            data[batch_size*b_idx+x_idx] = transformed_batch[x_idx].cpu().numpy()

class ColoredMNIST(torchvision.datasets.MNIST):
    input_shape = (3, 28, 28)
    num_classes = 2
    
    def __init__(self, *args, num_colors=4, normalize=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        ColoredMNIST.num_classes = num_colors
        
        color_0 = np.array([0.0, 0.0, 1.0]).reshape((3, 1, 1)) # blue
        color_1 = np.array([1.0, 0.0, 0.0]).reshape((3, 1, 1)) # red
        colors = [color_0 + delta*(color_1-color_0) for delta in np.linspace(0, 1, num_colors)]
        
        self.orig_data = len(self.data)*[np.zeros((1, 28, 28), dtype=np.float)]
        self.new_data = len(self.data)*[np.zeros((1, 28, 28), dtype=np.float)]
        self.color_targets = len(self.targets)*[0]
        for idx, data in enumerate(self.data):
            data = np.array(data).astype(np.float)
            color_target = np.random.randint(num_colors)
            color = colors[color_target]
            colored_data = np.stack(
                [channel*(data.squeeze()) for channel in color],
                axis=0
            )
            if normalize:
                data = normalize_tensor(data)
                colored_data = normalize_tensor(colored_data)
            self.orig_data[idx] = data
            self.new_data[idx] = colored_data
            self.color_targets[idx] = color_target
    
    def __getitem__(self, idx):
        _, target = super().__getitem__(idx)
        orig_data = self.orig_data[idx]
        orig_data = torch.from_numpy(np.array(orig_data)).to(torch.float)
        data = torch.from_numpy(self.new_data[idx]).to(torch.float).reshape(ColoredMNIST.input_shape)
        if self.transform is not None:
            data = self.transform(data)
        color_target = torch.tensor(self.color_targets[idx]).to(torch.long)
        return data, color_target, {'orig_image': orig_data, 'target': target}

class WatermarkedMNIST(torchvision.datasets.MNIST):
    input_shape = (1, 28, 28)
    num_classes = 2
    
    def __init__(self, *args, deterministic_position=False, deterministic_radius=False, normalize=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.orig_data = len(self.data)*[np.zeros((1, 28, 28), dtype=np.float)]
        self.new_data = len(self.data)*[np.zeros((1, 28, 28), dtype=np.float)]
        self.watermark_targets = len(self.targets)*[0]
        for idx, data in enumerate(self.data):
            data = np.array(data).astype(np.float)
            data = normalize_tensor(data)
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
            self.new_data[idx] = watermarked_data
            self.watermark_targets[idx] = watermark_target
        
    def __getitem__(self, idx):
        _, target = super().__getitem__(idx)
        orig_data = self.orig_data[idx]
        orig_data = torch.from_numpy(np.array(orig_data)).to(torch.float)
        data = torch.from_numpy(self.new_data[idx]).to(torch.float).reshape(WatermarkedMNIST.input_shape)
        if self.transform is not None:
            data = self.transform(data)
        watermark_target = torch.tensor(self.watermark_targets[idx]).to(torch.long)
        return data, watermark_target, {'orig_image': orig_data, 'target': target}