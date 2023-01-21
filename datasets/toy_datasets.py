import numpy as np
from numpy.random import default_rng
import torch
from torch.utils.data import Dataset

class GaussianDataset(Dataset):
    
    rng = None
    classes = None
    useful_features = None
    spurious_features = None
    useful_means = None
    useful_stds = None
    spurious_means = None
    spurious_stds = None
    
    def __init__(self,
                 classes=2,
                 useful_features=2,
                 spurious_features=98,
                 samples=1000,
                 transform=None,
                 target_transform=None,
                 use_existing_distribution=True,
                 seed=0,
                 std_rescale=1.0,
                 unit_std=False,
                 **kwargs
                 ):
        super().__init__()
        
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.rng = default_rng()
        
        existing_distribution_exists = (
            GaussianDataset.rng is not None and
            GaussianDataset.classes is not None and
            GaussianDataset.useful_features is not None and
            GaussianDataset.spurious_features is not None and
            GaussianDataset.useful_means is not None and
            GaussianDataset.useful_stds is not None and
            (spurious_features==0 or GaussianDataset.spurious_means is not None) and
            (spurious_features==0 or GaussianDataset.spurious_stds is not None))
        if not(use_existing_distribution) or not(existing_distribution_exists):
            GaussianDataset.rng = default_rng(seed)
            GaussianDataset.classes = classes
            GaussianDataset.useful_features = useful_features
            GaussianDataset.spurious_features = spurious_features
            GaussianDataset.useful_means = GaussianDataset.rng.standard_normal((classes, useful_features))
            for c in range(GaussianDataset.classes):
                GaussianDataset.useful_means[c, :] += c
            GaussianDataset.useful_stds = std_rescale*GaussianDataset.rng.gamma(2, 2, (classes, useful_features))/4
            if unit_std:
                GaussianDataset.useful_stds = np.ones_like(GaussianDataset.useful_stds)
            if GaussianDataset.spurious_features > 0:
                GaussianDataset.spurious_means = GaussianDataset.rng.standard_normal((GaussianDataset.spurious_features,))
                GaussianDataset.spurious_stds = std_rescale*GaussianDataset.rng.gamma(2, 2, (GaussianDataset.spurious_features,))/4
                if unit_std:
                    GaussianDataset.spurious_stds = np.ones_like(GaussianDataset.spurious_stds)
        
        self.x, self.y = [], []
        for c in range(GaussianDataset.classes):
            for n in range(self.samples//GaussianDataset.classes):
                features = [GaussianDataset.rng.normal(
                    GaussianDataset.useful_means[c],
                    GaussianDataset.useful_stds[c],
                    (GaussianDataset.useful_features,))]
                if GaussianDataset.spurious_features > 0:
                    features.append(self.rng.normal(
                        GaussianDataset.spurious_means,
                        GaussianDataset.spurious_stds,
                        (GaussianDataset.spurious_features,)))
                sample = np.concatenate(features, axis=-1)
                self.x.append(sample)
                self.y.append(c)
        self.x = np.stack(self.x)
        self.y = np.stack(self.y)
        self.x_shape = self.x[0].shape
        self.y_shape = self.y[0].shape
        assert len(self.x) == len(self.y)
        
    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).to(torch.float)
        y = torch.tensor(self.y[idx]).to(torch.long)
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y, {}
    
    def __len__(self):
        return len(self.x)
    
    def __repr__(self):
        s = 'Toy Gaussian dataset:'
        s += '\n\tClasses: {}'.format(GaussianDataset.classes)
        s += '\n\tUseful features: {}'.format(GaussianDataset.useful_features)
        s += '\n\tUseful means: {}'.format(GaussianDataset.useful_means)
        s += '\n\tUseful std. devs.: {}'.format(GaussianDataset.useful_stds)
        s += '\n\tSpurious features: {}'.format(GaussianDataset.spurious_features)
        s += '\n\tSpurious means: {}'.format(GaussianDataset.spurious_means)
        s += '\n\tSpurious std. devs.: {}'.format(GaussianDataset.spurious_stds)
        s += '\n\tTransform: {}'.format(self.transform)
        s += '\n\tTarget transform: {}'.format(self.target_transform)
        return s