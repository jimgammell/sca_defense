from copy import deepcopy
import os
import json
import time
from itertools import product
import torch
from torch import nn, optim

from datasets import google_power_traces, purdue_power_traces
import trials
import models
from utils import get_package_modules, get_package_module_names, list_module_attributes, get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))

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
    valid_arguments = {'dataset': [name for f in [google_power_traces, purdue_power_traces]
                                   for name in list_module_attributes(f)],
                       'device': ['cpu', 'cuda'] +\
                                 ['cuda:%d'%(idx) for idx in range(torch.cuda.device_count())],
                       'trial': get_package_module_names(trials)[0]}
    for suffix in ['', 'd', 'g']:
        def add_suffix(base):
            if suffix != '':
                return '_'.join((base, suffix))
            else:
                return base
        valid_types.update({add_suffix('model'): str,
                            add_suffix('model_kwargs'): dict,
                            add_suffix('loss_fn'): str,
                            add_suffix('loss_fn_kwargs'): dict,
                            add_suffix('optimizer'): str,
                            add_suffix('optimizer_kwargs'): dict})
        valid_arguments.update({add_suffix('model'): [name for file in get_package_modules(models)
                                                      for name in list_module_attributes(file)],
                                add_suffix('loss_fn'): [name for name in list_module_attributes(nn)],
                                add_suffix('optimizer'): [name for name in list_module_attributes(optim)]})
    return valid_types, valid_arguments

def expand_config(config):
    expanded_configs = []
    keys_to_expand = []
    for key, item in config.items():
        if 'expand' in key:
            keys_to_expand.append(key)
    for elements in product(*[config[key] for key in keys_to_expand]):
        expanded_config = deepcopy(config)
        for exp_key, settings in zip(keys_to_expand, elements):
            del expanded_config[exp_key]
            for key, item in settings.items():
                expanded_config.update({key: item})
        expanded_configs.append(expanded_config)
    return expanded_configs, keys_to_expand

def validate_config(config):
    valid_types, valid_arguments = get_config_valid_arguments()
    for key, item in config.items():
        if not key in valid_types.keys():
            print('Key {} is not in the list of valid keys {}'.format(key, valid_types.keys()))
            assert False
        if type(item) != valid_types[key]:
            print('Item {} corresponding to key {} has type {}, but should be of type {}'.format(item, key, valid_types[key]))
            assert False
        if key in valid_arguments.keys() and not(item in valid_arguments[key]):
            print('Item {} corresponding to key {} is not one of the valid arguments {}'.format(item, key, valid_arguments[key]))
            assert False
    
    for key, item in config.items():
        assert key in valid_types.keys()
        assert type(item) == valid_types[key]
        if key in valid_arguments.keys():
            assert item in valid_arguments[key]
    
def parse_config(path):
    config = load_config(path)
    expanded_configs, expanded_keys = expand_config(config)
    return expanded_configs, expanded_keys