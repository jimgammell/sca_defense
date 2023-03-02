import itertools
from copy import deepcopy
import os
import torch
from torch import nn, optim
from run_adversarial_trial import run_trial, plot_results

def unwrap_config_dict(config_dict):
    unwrapped_dicts = []
    keys, lists = [], []
    for key, item in config_dict.items():
        keys.append(key)
        lists.append(item)
    for element in itertools.product(*lists):
        unwrapped_dicts.append({key: item for key, item in zip(config_dict.keys(), element)})
    return unwrapped_dicts

def main():
    save_dir = os.path.join('.', 'results', 'adv_training_grid_search__critic')
    default_args = {
        'batch_size': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'disc_pretrain_epochs': 20,
        'gen_pretrain_epochs': 20,
        'ac_pretrain_epochs': 50,
        'train_epochs': 5,
        'eval_disc_posttrain_epochs': 20,
        'disc_sn': True,
        'gen_sn': True,
        'gen_loss_fn': nn.L1Loss(),
        'ce_eps': 1e-5,
        'ce_warmup_iter': 100,
        'ce_max_iter': 1000,
        'ce_opt': optim.Adam,
        'ce_opt_kwargs': {'lr': 1e-2},
        'project_rec_updates': True,
        'loss_mixture_coefficient': 0.1,
        'ind_eval_disc': True,
        'disc_orig_sample_prob': 0.0,
        'report_to_wandb': False,
        'pretrain_dir': os.path.join(save_dir, 'pretrained_models')
    }
    args_to_sweep = {
        'critic_method': ['gan', 'autoencoder', 'pixel_space'],
        'gen_loss_fn': [nn.L1Loss(), nn.MSELoss()],
        'ce_warmup_iter': [0, 1, 10],
        'ce_max_iter': [1, 10, 100],
        'ce_opt': [optim.Adam, optim.SGD],
        'project_rec_updates': [True, False],
        'gen_opt_kwargs': [{'lr': 0.01}, {'lr': 0.1}, {'lr': 0.1, 'momentum': 0.5}, {'lr': 0.01, 'momentum': 0.5}]
    }
    for trial_idx, sweep_config in enumerate(unwrap_config_dict(args_to_sweep)):
        try:
            args = deepcopy(default_args)
            args.update(sweep_config)
            trial_dir = os.path.join(save_dir, 'trial_{}'.format(trial_idx))
            args.update({'save_dir': trial_dir})
            print('Starting trial {}'.format(trial_idx))
            print('Configuration:')
            for key, item in args.items():
                print('\t{}: {}'.format(key, item))
            print('\n\n')
            run_trial(trial_info=args, **args)
            print('Done; plotting results.')
            plot_results(args['save_dir'])
            print('Done.')
            print('\n\n')
        except BaseException as e:
            print('Trial {} failed.'.format(trial_idx))
            print('Sweep config: {}'.format(sweep_config))
            print('Exception: {}'.format(e))
    
if __name__ == '__main__':
    main()