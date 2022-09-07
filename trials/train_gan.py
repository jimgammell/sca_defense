import os
import pickle
import time
from copy import copy
import numpy as np
import torch
from torch import nn, optim, distributions
from torch.utils.data import DataLoader

import models
import datasets
from trials.gan_common import GanExperiment
from utils import set_seed, get_attribute_from_package, get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))

def save_results(results, trial_dir):
    with open(os.path.join(trial_dir, 'training_metrics.pickle'), 'wb') as F:
        pickle.dump(results['results'], F)
    torch.save(results['trained_discriminator'].state_dict(),
               os.path.join(trial_dir, 'trained_discriminator'))
    torch.save(results['trained_generator'].state_dict(),
               os.path.join(trial_dir, 'trained_generator'))
    torch.save(results['discriminator_optimizer'].state_dict(),
               os.path.join(trial_dir, 'discriminator_optimizer'))
    torch.save(results['generator_optimizer'].state_dict(),
               os.path.join(trial_dir, 'generator_optimizer'))

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
    req('disc', str)
    req('gen', str)
    opt('disc_kwargs', {})
    opt('gen_kwargs', {})
    req('disc_loss_fn', str)
    req('gen_loss_fn', str)
    opt('disc_loss_fn_kwargs', {})
    opt('gen_loss_fn_kwargs', {})
    req('disc_opt', str)
    req('gen_opt', str)
    opt('disc_opt_kwargs', {})
    opt('gen_opt_kwargs', {})
    req('latent_var_shape', list)
    req('latent_var_distr', str)
    opt('latent_var_distr_kwargs', {})
    opt('conditions', [])
    opt('objective_formulation', 'Goodfellow')
    opt('disc_weight_clip_value', 0.01)
    req('dataset', str)
    opt('dataset_kwargs', {})
    opt('trial_kwargs', {'n_epochs': ['dg', 1], 'observe_gen_period': 1})
    opt('seed', time.time_ns()&0xFFFFFFFF)
    opt('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            
    return config_kwargs

def main(debug=False, **config_kwargs):
    print('Parsing config arguments...')    
    disc_name = config_kwargs['disc']
    gen_name = config_kwargs['gen']
    disc_kwargs = config_kwargs['disc_kwargs']
    gen_kwargs = config_kwargs['gen_kwargs']
    disc_loss_fn_name = config_kwargs['disc_loss_fn']
    gen_loss_fn_name = config_kwargs['gen_loss_fn']
    disc_loss_fn_kwargs = config_kwargs['disc_loss_fn_kwargs']
    gen_loss_fn_kwargs = config_kwargs['gen_loss_fn_kwargs']
    disc_opt_name = config_kwargs['disc_opt']
    gen_opt_name = config_kwargs['gen_opt']
    disc_opt_kwargs = config_kwargs['disc_opt_kwargs']
    gen_opt_kwargs = config_kwargs['gen_opt_kwargs']
    latent_var_shape = config_kwargs['latent_var_shape']
    latent_var_distr_name = config_kwargs['latent_var_distr']
    latent_var_distr_kwargs = copy(config_kwargs['latent_var_distr_kwargs'])
    conditions = config_kwargs['conditions']
    objective_formulation = config_kwargs['objective_formulation']
    disc_weight_clip_value = config_kwargs['disc_weight_clip_value']
    dataset_name = config_kwargs['dataset']
    dataset_kwargs = config_kwargs['dataset_kwargs']
    dataloader_kwargs = config_kwargs['dataloader_kwargs']
    trial_kwargs = config_kwargs['trial_kwargs']
    seed = config_kwargs['seed']
    device = config_kwargs['device']
    for key, item in copy(latent_var_distr_kwargs).items():
        if '__EXPAND' in key:
            del latent_var_distr_kwargs[key]
            latent_var_distr_kwargs.update({
                key.split('__EXPAND')[0]: torch.ones(latent_var_shape)*item})
    
    disc_constructor = get_attribute_from_package(disc_name, models)
    gen_constructor = get_attribute_from_package(gen_name, models)
    disc_loss_fn_constructor = getattr(nn, disc_loss_fn_name)
    gen_loss_fn_constructor = getattr(nn, gen_loss_fn_name)
    disc_opt_constructor = getattr(optim, disc_opt_name)
    gen_opt_constructor = getattr(optim, gen_opt_name)
    latent_var_distr_constructor = get_attribute_from_package(latent_var_distr_name, distributions)
    dataset_constructor = get_attribute_from_package(dataset_name, datasets)
    
    train_dataset = dataset_constructor(train=True, **dataset_kwargs)
    print('Training dataset:\n{}\n'.format(train_dataset))
    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    print('Training dataloader:\n{}\n'.format(train_dataloader))
    test_dataset = dataset_constructor(train=False, **dataset_kwargs)
    print('Test dataset:\n{}\n'.format(test_dataset))
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)
    print('Test dataloader:\n{}\n'.format(test_dataloader))
    eg_input = next(iter(train_dataloader))[0]
    discriminator = disc_constructor(eg_input.shape, conditions=conditions, **disc_kwargs)
    print('Discriminator:\n{}\n'.format(discriminator))
    generator = gen_constructor(np.prod(latent_var_shape[1:]), eg_input.shape, conditions=conditions, **gen_kwargs)
    print('Generator:\n{}\n'.format(generator))
    disc_loss_fn = disc_loss_fn_constructor(**disc_loss_fn_kwargs)
    print('Discriminator loss function:\n{}\n'.format(disc_loss_fn))
    gen_loss_fn = gen_loss_fn_constructor(**gen_loss_fn_kwargs)
    print('Generator loss function:\n{}\n'.format(gen_loss_fn))
    disc_opt = disc_opt_constructor(discriminator.parameters(), **disc_opt_kwargs)
    print('Discriminator optimizer:\n{}\n'.format(disc_opt))
    gen_opt = gen_opt_constructor(generator.parameters(), **gen_opt_kwargs)
    print('Generator optimizer:\n{}\n'.format(gen_opt))
    latent_var_distr = latent_var_distr_constructor(**latent_var_distr_kwargs)
    print('Latent variable distribution:\n{}\n'.format(latent_var_distr))
    set_seed(seed)
    print('Seed: {}'.format(seed))
    
    gan_trainer = GanExperiment(discriminator,
                                generator,
                                disc_loss_fn,
                                gen_loss_fn,
                                disc_opt,
                                gen_opt,
                                latent_var_distr,
                                device,
                                conditions,
                                objective_formulation,
                                disc_weight_clip_value)
    
    if debug:
        n_epochs = [['dg', 1]]
    else:
        n_epochs = trial_kwargs['n_epochs']
    n_epochs[0][1] += 1
    observe_gen_period = trial_kwargs['observe_gen_period']
    disc_steps_per_gen_step = trial_kwargs['disc_steps_per_gen_step']
    epoch_idx, sub_trial_idx = 0, 0
    def get_sub_trial_key(_sub_trial_idx=None):
        if _sub_trial_idx == None:
            _sub_trial_idx = sub_trial_idx
        return (_sub_trial_idx, *n_epochs[_sub_trial_idx])
    Results = {get_sub_trial_key(_sub_trial_idx): {epoch_idx: {}
                               for epoch_idx in range(sub_trial_epochs[1])}
               for _sub_trial_idx, sub_trial_epochs in enumerate(n_epochs)}
    def eval_models(update_indices=True):
        nonlocal epoch_idx, sub_trial_idx
        results = gan_trainer.eval_epoch(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
        Results[get_sub_trial_key()][epoch_idx].update(results)
        if update_indices:
            epoch_idx += 1
            if epoch_idx >= n_epochs[sub_trial_idx][1]:
                epoch_idx = 0
                sub_trial_idx += 1
    print('Evaluating initial performance...')
    eval_models()
    while sub_trial_idx < len(n_epochs):
        print('Training epoch {}/{} of sub-trial {}/{}...'.format(epoch_idx+1, n_epochs[sub_trial_idx][1],
                                                                  sub_trial_idx+1, len(n_epochs)))
        gan_trainer.train_epoch(train_dataloader,
                                train_gen='g' in n_epochs[sub_trial_idx][0],
                                train_disc='d' in n_epochs[sub_trial_idx][0],
                                disc_steps_per_gen_step=disc_steps_per_gen_step)
        print('Evaluating epoch {}/{} of sub-trial {}/{}...'.format(epoch_idx+1, n_epochs[sub_trial_idx][1],
                                                                    sub_trial_idx+1, len(n_epochs)))
        eval_models()
    
    return {'results': Results,
            'trained_discriminator': discriminator,
            'trained_generator': generator,
            'discriminator_optimizer': disc_opt,
            'generator_optimizer': gen_opt}