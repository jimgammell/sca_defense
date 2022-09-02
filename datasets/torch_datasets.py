import os
import torchvision.datasets

from datasets.common import preprocess_transform

class MNIST(torchvision.datasets.MNIST):
    def __init__(self, 
                 *args, 
                 transform=None, 
                 transform_kwargs={},
                 **kwargs):
        transform = preprocess_transform(transform, transform_kwargs)
        super().__init__(*args,
                         transform=transform,
                         root=os.path.join('.', 'saved_datasets', 'MNIST'),
                         download=True,
                         **kwargs)