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
    save_dir = os.path.join('.', 'results', 'gan_gridsearch_xx')
    if os.path.exists(save_dir):
        if overwrite:
            shutil.rmtree(save_dir)
            base_trial_idx = 0
        else:
            files = os.listdir(save_dir)
            trial_indices = [
                int(f.split('_')[1]) for f in files if f.split('_')[0] == 'trial'
            ]
            if len(trial_indices) > 0:
                base_trial_idx = np.max(trial_indices)+1
            else:
                base_trial_idx = 0
    else:
        base_trial_idx = 0
    default_args = {
        'save_dir': save_dir,
    }
    n_repetitions = 2
    args_to_sweep = {
        'dataset': [ColoredMNIST, WatermarkedMNIST],
        'gen_leakage_coefficient': [0.5],
        'y_clamp': [None, 0],
        'gen_leakage_ramp_duration': [0.0, 0.25],
        'disc_gradient_penalty': [100.0, 0.0],
        'l1_rec_coefficient': [0.0],
        'mixup_alpha': [1.0],
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
                if args['dataset'] == ColoredMNIST:
                    args['average_deviation_penalty'] = 1e0
                elif args['dataset'] == WatermarkedMNIST:
                    args['average_deviation_penalty'] = 1e-1
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
                rep_config['average_deviation_penalty'] = args['average_deviation_penalty']
                with open(os.path.join(trial_dir, 'sweep_config.json'), 'w') as F:
                    json.dump(rep_config, F)
        except Exception:
            traceback.print_exc()
    
if __name__ == '__main__':
    overwrite = '--overwrite' in sys.argv
    main(overwrite=overwrite)