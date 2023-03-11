import itertools
from copy import deepcopy
import os
import torch
from torch import nn, optim
from datasets.classified_mnist import ColoredMNIST, WatermarkedMNIST
from gan_trial import run_trial, generate_animation, plot_traces
from gan_train import hinge_loss, leaky_hinge_loss, sum_loss

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
    save_dir = os.path.join('.', 'results', 'gan_gridsearch')
    default_args = {
        'save_dir': save_dir,
    }
    args_to_sweep = {
        'dataset': [WatermarkedMNIST, ColoredMNIST],
        'gen_loss': [hinge_loss, leaky_hinge_loss, sum_loss],
        'disc_loss': [hinge_loss, leaky_hinge_loss],
        'gen_kwargs': [
            {'num_kernels': 16, 'bottleneck_width': 64, 'use_instance_norm': True},
            {'num_kernels': 16, 'bottleneck_width': 64, 'use_spectral_norm': True}
        ],
        'disc_kwargs': [
            {'num_kernels': 16, 'use_spectral_norm': True},
            {'num_kernels': 16, 'use_instance_norm': True}
        ]
    }
    for trial_idx, sweep_config in enumerate(unwrap_config_dict(args_to_sweep)):
        #try:
        if True:
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
        #except BaseException as e:
        #    print('Trial {} failed.'.format(trial_idx))
        #    print('Sweep config: {}'.format(sweep_config))
        #    print('Exception: {}'.format(e))
    
if __name__ == '__main__':
    main()