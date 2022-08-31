import os
import pickle
import time
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

import models
import datasets
from utils import set_seed, get_attribute_from_package, get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))
from trials.single_model_common import train_epoch, eval_epoch

def save_results(results, trial_dir):
    with open(os.path.join(trial_dir, 'training_metrics.pickle'), 'wb') as F:
        pickle.dump(results['results'], F)
    torch.save(results['trained_model'].state_dict(),
               os.path.join(trial_dir, 'trained_model_state'))
    torch.save(results['optimizer'].state_dict(),
               os.path.join(trial_dir, 'optimizer_state'))

def preprocess_config(config_kwargs):
    def req(key, valid_type=None):
        assert key in config_kwargs.keys()
        if valid_type != None:
            assert type(config_kwargs[key]) == valid_type
    def opt(key, default):
        if not key in config_kwargs.keys():
            config_kwargs[key] = default
        else:
            assert type(config_kwargs[key]) == type(default)
    req('model', str)
    opt('model_kwargs', {})
    req('loss_fn', str)
    opt('loss_fn_kwargs', {})
    req('optimizer', str)
    opt('optimizer_kwargs', {})
    req('dataset', str)
    opt('dataset_kwargs', {})
    opt('trial_kwargs', {'n_epochs': 1})
    opt('seed', time.time_ns()&0xFFFFFFFF)
    opt('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    return config_kwargs
    
def main(debug=False, **config_kwargs):
    print('Parsing config arguments...')
    model_name = config_kwargs['model']
    model_kwargs = config_kwargs['model_kwargs']
    loss_fn_name = config_kwargs['loss_fn']
    loss_fn_kwargs = config_kwargs['loss_fn_kwargs']
    optimizer_name = config_kwargs['optimizer']
    optimizer_kwargs = config_kwargs['optimizer_kwargs']
    dataset_name = config_kwargs['dataset']
    dataset_kwargs = config_kwargs['dataset_kwargs']
    trial_kwargs = config_kwargs['trial_kwargs']
    seed = config_kwargs['seed']
    device = config_kwargs['device']
    
    model_constructor = get_attribute_from_package(model_name, models)
    loss_fn_constructor = getattr(nn, loss_fn_name)
    optimizer_constructor = getattr(optim, optimizer_name)
    dataset_constructor = get_attribute_from_package(dataset_name, datasets)
    
    train_dataset = dataset_constructor(train=True, **dataset_kwargs)
    print('Training dataset:\n{}\n'.format(train_dataset))
    test_dataset = dataset_constructor(train=False, **dataset_kwargs)
    print('Testing dataset:\n{}\n'.format(test_dataset))
    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    print('Training dataloader:\n{}\n'.format(train_dataloader))
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)
    print('Testing dataloader:\n{}\n'.format(test_dataloader))
    eg_input = next(iter(train_dataloader))[0]
    discriminator = model_constructor(eg_input.shape, **model_kwargs).to(device)
    print('Discriminator:\n{}\n'.format(discriminator))
    loss_fn = loss_fn_constructor(**loss_fn_kwargs)
    print('Loss function:\n{}\n'.format(loss_fn))
    optimizer = optimizer_constructor(discriminator.parameters(), **optimizer_kwargs)
    print('Optimizer:\n{}\n'.format(optimizer))
    set_seed(seed)
    print('Seed: {}'.format(seed))
    
    if debug:
        n_epochs = 1
    else:
        n_epochs = trial_kwargs['n_epochs']
    Results = {epoch_idx: {} for epoch_idx in range(n_epochs+1)}
    epoch_idx = 0
    def eval_model():
        nonlocal epoch_idx
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
    print('Evaluating initial performance...')
    eval_model()
    while epoch_idx <= n_epochs:
        print('Training epoch {}...'.format(epoch_idx))
        train_epoch(train_dataloader,
                    discriminator,
                    optimizer,
                    loss_fn,
                    device)
        print('Evaluating epoch {}...'.format(epoch_idx))
        eval_model()
    
    return {'results': Results,
            'trained_model': discriminator,
            'optimizer': optimizer}