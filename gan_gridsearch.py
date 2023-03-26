import itertools
from copy import deepcopy
import os
import torch
import json
import numpy as np
from torch import nn, optim
from datasets.classified_mnist import ColoredMNIST, WatermarkedMNIST
from gan_trial import run_trial, generate_animation, plot_traces

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
    save_dir = os.path.join('.', 'results', 'gan_gridsearch_ix')
    default_args = {
        'save_dir': save_dir,
    }
    n_repetitions = 3
    args_to_sweep = {
        'dataset': [ColoredMNIST, WatermarkedMNIST],
        'clip_gradients': [False, True],
        'whiten_features': [False],
        'disc_invariance_coefficient': [0.0],
        'disc_leakage_coefficient': [0.5],
        'gen_leakage_coefficient': [0.0, 0.01, 0.1, 0.5, 0.9, 0.99],
        'disc_steps_per_gen_step': [5.0]
    }
    for trial_idx, sweep_config in enumerate(unwrap_config_dict(args_to_sweep)):
        try:
            for repetition in range(n_repetitions):
                args = deepcopy(default_args)
                args.update(sweep_config)
                trial_dir = os.path.join(save_dir, 'trial_{}__rep_{}'.format(trial_idx, repetition))
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
                rep_config = {key: str(item) for key, item in sweep_config.items()}
                with open(os.path.join(trial_dir, 'sweep_config.json'), 'w') as F:
                    json.dump(rep_config, F)
        except BaseException as e:
            print(e)
    
if __name__ == '__main__':
    main()