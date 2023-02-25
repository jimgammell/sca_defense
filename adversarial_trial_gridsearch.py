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
    save_dir = os.path.join('.', 'results', 'adv_training_grid_search')
    default_args = {
        'batch_size': 512,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'disc_pretrain_epochs': 20,
        'gen_pretrain_epochs': 20,
        'train_epochs': 20,
        'eval_disc_posttrain_epochs': 20,
        'disc_sn': True,
        'gen_sn': True,
        'gen_loss_fn': nn.MSELoss(),
        'ce_eps': 1e-5,
        'ce_warmup_iter': 100,
        'ce_max_iter': 1000,
        'ce_opt': optim.Adam,
        'ce_opt_kwargs': {'lr': 1e-2},
        'project_rec_updates': True,
        'loss_mixture_coefficient': 0.01,
        'ind_eval_disc': True,
        'disc_orig_sample_prob': 0.0,
        'report_to_wandb': False
    }
    args_to_sweep = {
        'disc_sn': [True, False],
        'gen_sn': [True, False],
        'gen_loss_fn': [nn.MSELoss(), nn.L1Loss()],
        'project_rec_updates': [True, False],
        'loss_mixture_coefficient': [0.01, 0.1, 0.5]
    }
    for trial_idx, sweep_config in enumerate(unwrap_config_dict(args_to_sweep)):
        args = deepcopy(default_args)
        args.update(sweep_config)
        trial_dir = os.path.join(save_dir, 'trial_{}'.format(trial_idx))
        args.update({'save_dir': trial_dir})
        print('Starting trial {}'.format(trial_idx))
        print('Configuration:')
        for key, item in args.items():
            print('\t{}: {}'.format(key, item))
        print('\n\n')
        run_trial(**args)
        print('Done; plotting results.')
        plot_results(args['save_dir'])
        print('Done.')
        print('\n\n')
    
if __name__ == '__main__':
    main()