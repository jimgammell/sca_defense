import pickle
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import models
import datasets
from utils import set_seed, get_attribute_from_package
from trials.single_model_common import train_epoch, eval_epoch

def save_results(results, trial_dir):
    with open(os.path.join(trial_dir, 'training_metrics.pickle'), 'wb') as F:
        pickle.dump(results['results'], F)
    torch.save(results['trained_model'].state_dict(), os.path.join(trial_dir, 'trained_model_state'))
    torch.save(results['optimizer'].state_dict(), os.path.join(trial_dir, 'optimizer_state'))

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
    optimizer = optimizer_constructor(discriminator.parameters(), **optimizer_constructor_kwargs)
    train_dataset = dataset_constructor(train=True, **dataset_kwargs)
    test_dataset = dataset_constructor(train=False, **dataset_kwargs)
    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)
    set_seed(seed)
    
    n_epochs = trial_kwargs['n_epochs']
    Results = {epoch_idx: {} for epoch_idx in range(n_epochs+1)}
    epoch_idx = 0
    def eval_model():
        training_results = eval_epoch(train_dataloader,
                                      discriminator,
                                      loss_fn,
                                      device)
        test_results = eval_epoch(test_dataloader,
                                  discriminator,
                                  loss_fn,
                                  device)
        Results[epoch_idx].update({'train': training_results,
                                   'test': test_results})
        epoch_idx += 1
    eval_model()
    while epoch_idx <= n_epochs:
        train_epoch(train_dataloader,
                    discriminator,
                    loss_fn,
                    device)
        eval_model()
    
    return {'results': Results,
            'trained_model': discriminator,
            'optimizer': optimizer}