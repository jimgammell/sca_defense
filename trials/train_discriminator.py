from torch.utils.data import DataLoader
from torch import nn, optim

import models
import datasets
from utils import set_seed, get_attribute_from_package

def save_results(*args, **kwargs):
    pass

def main(model_name, model_kwargs,
         loss_fn_name, loss_fn_kwargs,
         optimizer_name, optimizer_kwargs,
         dataset_name, dataset_kwargs,
         dataloader_kwargs,
         trial_kwargs,
         seed,
         device):
    model_constructor = get_attribute_from_package(models, model_name)
    loss_fn_constructor = getattr(nn, loss_fn_name)
    optimizer_constructor = getattr(optim, optimizer_name)
    dataset_constructor = get_attribute_from_package(datasets, dataset_name)
    
    discriminator = model_constructor(**model_kwargs).to(device)
    loss_fn = loss_fn_constructor(**loss_fn_kwargs)
    optimizer = optimizer_constructor(**optimizer_constructor_kwargs)
    train_dataset = dataset_constructor(train=True, **dataset_kwargs)
    test_dataset = dataset_constructor(train=False, **dataset_kwargs)
    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)
    set_seed(seed)