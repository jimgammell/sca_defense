import itertools
from copy import deepcopy
import os
import torch
from torch import nn, optim
from lagrangian_trial import run_trial, generate_animation, plot_traces

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
    save_dir = os.path.join('.', 'results', 'lagrangian_trial')
    default_args = {
        'save_dir': save_dir,
        'pretrain_dir': os.path.join(save_dir, 'pretrained_models')
    }
    args_to_sweep = {
        'rec_loss_fn': [nn.MSELoss, nn.L1Loss],
        'lbd_opt_kwargs': [{'lr': 1e-1}, {'lr': 1e0}, {'lr': 1e1}],
        'separate_cls_partition': [True, False]
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
            plot_traces(trial_dir)
            generate_animation(trial_dir)
            print('Done.')
            print('\n\n')
        except BaseException as e:
            print('Trial {} failed.'.format(trial_idx))
            print('Sweep config: {}'.format(sweep_config))
            print('Exception: {}'.format(e))
    
if __name__ == '__main__':
    main()