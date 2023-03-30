import itertools
from copy import deepcopy
import os
import shutil
import sys
import traceback
import random
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

def main(overwrite=False):
    save_dir = os.path.join('.', 'results', 'gan_gridsearch_xii')
    if os.path.exists(save_dir):
        if overwrite:
            shutil.rmtree(save_dir)
            base_trial_idx = 0
        else:
            files = os.listdir(save_dir)
            trial_indices = [
                int(f.split('_')[1]) for f in files if f.split('_')[0] == 'trial'
            ]
            base_trial_idx = np.max(trial_indices)+1
    else:
        base_trial_idx = 0
    default_args = {
        'save_dir': save_dir,
    }
    n_repetitions = 3
    args_to_sweep = {
        'dataset': [ColoredMNIST],
        'clip_gradients': [False],
        'whiten_features': [False],
        'disc_invariance_coefficient': [1e2],
        'disc_leakage_coefficient': [0.5],
        'gen_leakage_coefficient': [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
        'disc_steps_per_gen_step': [5.0]
    }
    for trial_idx, sweep_config in enumerate(unwrap_config_dict(args_to_sweep)):
        trial_idx += base_trial_idx
        try:
            for repetition in range(n_repetitions):
                random.seed(repetition)
                np.random.seed(repetition)
                torch.random.manual_seed(repetition)
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
        except Exception:
            traceback.print_exc()
    
if __name__ == '__main__':
    overwrite = '--overwrite' in sys.argv
    main(overwrite=overwrite)