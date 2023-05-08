import argparse
import random
import numpy as np
import torch
import itertools
from train_classifier_on_google_scaaml_dataset import main as run_trial

hparams = {
    'activation': ['gelu'],
    'norm_layer': ['instance_norm'],
    'kernel_size': [7],
    'block': ['convnext']
}

def unwrap_config_dict(config_dict):
    unwrapped_dicts = []
    keys, lists = [], []
    for key, item in config_dict.items():
        keys.append(key)
        lists.append(item)
    for element in itertools.product(*lists):
        unwrapped_dicts.append({key: item for key, item in zip(config_dict.keys(), element)})
    return unwrapped_dicts

def main(device=None):
    config_dicts = unwrap_config_dict(hparams)
    for config_dict in config_dicts:
        print('Starting trial with settings {}'.format(config_dict))
        print()
        run_trial(device=device, classifier_kwargs=config_dict)
        print()
        print('Done with trial.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int, help='Random seed to use for this trial.')
    parser.add_argument('--device', default=None, type=str, help='Device to use for this trial.')
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    main(device=args.device)