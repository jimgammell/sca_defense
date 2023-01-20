import numpy as np
from numpy.random import default_rng
import torch
from torch.utils.data import Dataset

class GaussianDataset(Dataset):
    def __init__(self,
                 classes=2,
                 useful_features=2,
                 spurious_features=98,
                 samples=1000,
                 transform=None,
                 target_transform=None
                 ):
        super().__init__()
        
        self.classes = classes
        self.useful_features = useful_features
        self.spurious_features = spurious_features
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        
        self.rng = default_rng()
        self.useful_means = self.rng.standard_normal((classes, useful_features))
        self.useful_stds = self.rng.gamma(2, 2, (classes, useful_features))/4
        if self.spurious_features > 0:
            self.spurious_means = self.rng.standard_normal((spurious_features,))
            self.spurious_stds = self.rng.gamma(2, 2, (spurious_features,))/4
        
        self.x, self.y = [], []
        for c in range(self.classes):
            for n in range(self.samples//self.classes):
                features = [self.rng.normal(self.useful_means[c], self.useful_stds[c], (self.useful_features,))]
                if self.spurious_features > 0:
                    features.append(self.rng.normal(self.spurious_means, self.spurious_stds, (self.spurious_features,)))
                sample = np.concatenate(features, axis=-1)
                self.x.append(sample)
                self.y.append(c)
        self.x = np.stack(self.x)
        self.y = np.stack(self.y)
        assert len(self.x) == len(self.y)
        
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y
    
    def __len__(self):
        return len(self.x)