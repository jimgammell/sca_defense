import itertools
from copy import deepcopy
import os
import torch
import json
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
    save_dir = os.path.join('.', 'results', 'gan_gridsearch_ii')
    default_args = {
        'save_dir': save_dir,
    }
    args_to_sweep = {
        'dataset': [ColoredMNIST],
        'gen_skip_connection': [True, False],
        'disc_steps_per_gen_step': [1.0, 5.0],
        'project_gen_updates': [False, True],
        'gen_leakage_coefficient': [0.1, 0.5, 0.9]
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
            for key, item in sweep_config.items():
                if type(item) != str:
                    sweep_config[key] = str(item)
            with open(os.path.join(trial_dir, 'sweep_config.json'), 'w') as F:
                json.dump(sweep_config, F)
        except:
            pass
    
if __name__ == '__main__':
    main()