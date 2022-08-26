from copy import deepcopy
import json
import time
from itertools import product
import torch
from torch import nn, optim

import datasets
import trials
import models

def load_config(path):
    with open(path, 'r') as F:
        config = json.load(F)
    return config

def get_config_valid_arguments():
    valid_types = {'trial_description': str,
                   'dataset': str,
                   'dataset_kwargs': dict,
                   'dataloader_kwargs': dict,
                   'device': str,
                   'seed': int,
                   'trial': str,
                   'trial_kwargs': dict}
    valid_arguments = {'dataset': [name for name in dir(file)
                                   for file in [datasets.google_power_traces, datasets.purdue_power_traces]],
                       'device': ['cpu', 'cuda'] +\
                                 ['cuda:%d'%(idx) for idx in range(torch.cuda.device_count())],
                       'trial': [name for name in dir(file)
                                 for file in dir(trials)]}
    for suffix in ['', 'd', 'g']:
        def add_suffix(base):
            if suffix != '':
                return '_'.join(base, suffix)
            else:
                return base
        valid_types.update({add_suffix('model'): str,
                            add_suffix('model_kwargs'): dict,
                            add_suffix('loss_fn'): str,
                            add_suffix('loss_fn_kwargs'): dict,
                            add_suffix('optimizer'): str,
                            add_suffix('optimizer_kwargs'): dict})
        valid_arguments.update({add_suffix('model'): [name for name in dir(file)
                                                      for file in dir(models)],
                                add_suffix('loss_fn'): [name for name in dir(nn)],
                                add_suffix('optimizer'): [name for name in dir(optim)]})
    return valid_types, valid_arguments

def apply_default_settings(config):
    default_arguments = {'trial_description': 'none provided',
                         'dataset_kwargs': {},
                         'dataloader_kwargs': {},
                         'trial_kwargs': {},
                         'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                         'seed': time.time_ns()&0xFFFFFFFF}
    if 'model' in config.keys():
        default_arguments.update({'model_kwargs': {},
                                  'loss_fn_kwargs': {},
                                  'optimizer': 'Adam',
                                  'optimizer_kwargs': {}})
    else:
        default_arguments.update({'model_kwargs_d': {},
                                  'model_kwargs_g': {},
                                  'loss_fn_kwargs_d': {},
                                  'loss_fn_kwargs_g': {},
                                  'optimizer_d': 'Adam',
                                  'optimizer_g': 'Adam',
                                  'optimizer_kwargs_d': {},
                                  'optimizer_kwargs_g': {}})
    for key, item in default_arguments.items():
        if not(key in config.keys()):
            config.update({key: item})
    return config

def expand_config(config):
    expanded_configs = []
    keys_to_expand = []
    valid_types, _ = get_config_valid_arguments()
    for key, item in config.items():
        if type(item) == list:
            if type(valid_types[key] != list):
                assert type(item[0]) == valid_types[key]
                keys_to_expand.append(key)
            else:
                if type(item[0]) == list:
                    keys_to_expand.append(key)
    for elements in product(*[config[key] for key in keys_to_expand]):
        expanded_config = deepcopy(config)
        for key, element in zip(keys_to_expand, elements):
            expanded_config[key] = element
        expanded_configs.append(expanded_config)
    return expanded_configs, keys_to_expand

def validate_config(config):
    valid_types, valid_arguments = get_config_valid_arguments()
    for key, item in config.items():
        assert key in valid_types.keys()
        assert type(item) == valid_types[key]
        if key in valid_arguments.keys():
            assert item in valid_arguments[key]
    
def parse_config(path):
    config = load_config(path)
    config = apply_default_settings(config)
    expanded_configs, expanded_keys = expand_config(config)
    for expanded_config in expanded_configs:
        validate_config(expanded_config)
    return expanded_configs, expanded_keys