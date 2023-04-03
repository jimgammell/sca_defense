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
    save_dir = os.path.join('.', 'results', 'gan_gridsearch_xvii')
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
    n_repetitions = 5
    args_to_sweep = {
        'dataset': [ColoredMNIST, WatermarkedMNIST],
        'y_clamp': [0],
        'l1_rec_coefficient': [0.0],
        'mixup_alpha': [1.0],
        'average_deviation_penalty': [1e0],
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