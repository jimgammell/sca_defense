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
from trials.antigan_common import AntiGanExperiment
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
    def opt(key, default, enforce_default_type=True):
        if not key in config_kwargs.keys():
            config_kwargs[key] = default
        elif enforce_default_type:
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
    opt('latent_var_shape', [32, 100])
    opt('latent_var_distr', None, enforce_default_type=False)
    opt('latent_var_distr_kwargs', {})
    opt('use_labels', True)
    opt('objective_formulation', 'Complement')
    opt('gen_weight_clamp', None, enforce_default_type=False)
    opt('disc_weight_clamp', None, enforce_default_type=False)
    opt('num_classes', 10)
    req('dataset', str)
    opt('dataset_kwargs', {})
    opt('dataloader_kwargs', {})
    trial_kwargs = {'n_epochs': [['dg', 1]],
                    'observe_gen_period': 1,
                    'gen_steps_per_disc_step': None,
                    'disc_steps_per_gen_step': None}
    if 'trial_kwargs' in config_kwargs.keys():
        trial_kwargs.update(config_kwargs['trial_kwargs'])
    config_kwargs['trial_kwargs'] = trial_kwargs
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
    use_labels = config_kwargs['use_labels']
    objective_formulation = config_kwargs['objective_formulation']
    gen_weight_clamp = config_kwargs['gen_weight_clamp']
    disc_weight_clamp = config_kwargs['disc_weight_clamp']
    num_classes = config_kwargs['num_classes']
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
    if latent_var_distr_name != None:
        latent_var_distr_constructor = get_attribute_from_package(latent_var_distr_name, distributions)
    dataset_constructor = get_attribute_from_package(dataset_name, datasets)
    
    train_dataset = dataset_constructor(train=True, **dataset_kwargs)
    print('Training dataset:\n{}\n'.format(train_dataset))
    train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
    print('Training dataloader:\n{}\n'.format(train_dataloader))
    test_dataset = dataset_constructor(train=False, **dataset_kwargs)
    print('Testing dataset:\n{}\n'.format(test_dataset))
    test_dataloader = DataLoader(test_dataset, **dataloader_kwargs)
    print('Testing dataloader:\n{}\n'.format(test_dataloader))
    eg_input = next(iter(train_dataloader))[0]
    discriminator = disc_constructor(eg_input.shape, **disc_kwargs)
    print('Discriminator:\n{}\n'.format(discriminator))
    generator = gen_constructor(latent_var_shape[-1], num_classes if use_labels else 0, eg_input.shape, **gen_kwargs)
    print('Generator:\n{}\n'.format(generator))
    disc_loss_fn = disc_loss_fn_constructor(**disc_loss_fn_kwargs)
    print('Discriminator loss function:\n{}\n'.format(disc_loss_fn))
    gen_loss_fn = gen_loss_fn_constructor(**gen_loss_fn_kwargs)
    print('Generator loss function:\n{}\n'.format(gen_loss_fn))
    disc_opt = disc_opt_constructor(discriminator.parameters(), **disc_opt_kwargs)
    print('Discriminator optimizer:\n{}\n'.format(disc_opt))
    gen_opt = gen_opt_constructor(generator.parameters(), **gen_opt_kwargs)
    print('Generator optimizer:\n{}\n'.format(gen_opt))
    if latent_var_distr_name != None:
        latent_var_distr = latent_var_distr_constructor(**latent_var_distr_kwargs)
    else:
        latent_var_distr = None
    print('Latent variable distribution:\n{}\n'.format(latent_var_distr))
    set_seed(seed)
    print('Seed: {}'.format(seed))
    
    antigan_trainer = AntiGanExperiment(discriminator,
                                        generator,
                                        disc_loss_fn,
                                        gen_loss_fn,
                                        disc_opt,
                                        gen_opt,
                                        device,
                                        num_classes,
                                        latent_var_distr,
                                        use_labels,
                                        objective_formulation,
                                        gen_weight_clamp,
                                        disc_weight_clamp)
    if debug:
        n_epochs = [['dg', 1]]
    else:
        n_epochs = trial_kwargs['n_epochs']
    n_epochs[0][1] += 1
    observe_gen_period = trial_kwargs['observe_gen_period'] if 'observe_gen_period' in trial_kwargs.keys() else 1
    train_ind_disc_period = trial_kwargs['train_ind_disc_period'] if 'train_ind_disc_period' in trial_kwargs.keys() else 10
    epoch_idx, sub_trial_idx = 0, 0
    def get_sub_trial_key(_sub_trial_idx=None):
        if _sub_trial_idx == None:
            _sub_trial_idx = sub_trial_idx
        return (_sub_trial_idx, *n_epochs[_sub_trial_idx])
    Results = {get_sub_trial_key(_sub_trial_idx): {epoch_idx: {} for epoch_idx in range(sub_trial_epochs[1])}
               for _sub_trial_idx, sub_trial_epochs in enumerate(n_epochs)}
    def eval_models(update_indices=True):
        nonlocal epoch_idx, sub_trial_idx
        results = antigan_trainer.eval_epoch(train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                             sample_gen_images = epoch_idx%observe_gen_period == 0,
                                             train_independent_discriminator = epoch_idx%train_ind_disc_period == 0,
                                             ind_disc_epochs=1 if debug else 20)
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
        if ('g' in n_epochs[sub_trial_idx][0]) and ('d' in n_epochs[sub_trial_idx][0]):
            antigan_trainer.train_epoch(train_dataloader,
                                        disc_steps_per_gen_step=trial_kwargs['disc_steps_per_gen_step'],
                                        gen_steps_per_disc_step=trial_kwargs['gen_steps_per_disc_step'])
        else:
            antigan_trainer.train_epoch(train_dataloader,
                                        train_gen='g' in n_epochs[sub_trial_idx][0],
                                        train_disc='d' in n_epochs[sub_trial_idx][0])
        print('Evaluating epoch {}/{} of sub-trial {}/{}...'.format(epoch_idx+1, n_epochs[sub_trial_idx][1],
                                                                    sub_trial_idx+1, len(n_epochs)))
        eval_models()
    
    return {'results': Results,
            'trained_discriminator': discriminator,
            'trained_generator': generator,
            'discriminator_optimizer': disc_opt,
            'generator_optimizer': gen_opt}