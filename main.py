import os
import argparse
import json
from copy import deepcopy
import importlib
import torch
from run_trial import run_trial

def get_json_args(json_filename):
    def getattr_from_file(attr_name):
        modules = []
        for package in ['datasets', 'models', 'training']:
            for module_filename in os.listdir(os.path.join('.', package)):
                if module_filename in ['__init__.py', '__pycache__', '.ipynb_checkpoints']:
                    continue
                module = importlib.import_module('.'+module_filename.split('.')[0], package)
                modules.append(module)
        modules.extend([torch.nn, torch.optim])
        for module in modules:
            try:
                attr = getattr(module, attr_name)
                return attr
            except:
                pass
        raise Exception('Could not find attribute called \'{}\' in the searched modules [\n\t{}]'.format(
            attr_name, ',\n\t'.join(str(module) for module in modules)))
        
    def preprocess_config(config):
        for cfg_key, cfg_item in deepcopy(config).items():
            if cfg_key == 'getattr':
                for gat_key, gat_item in cfg_item.items():
                    attr = getattr_from_file(gat_item)
                    assert gat_key not in config.keys()
                    config[gat_key] = attr
                del config['getattr']
            elif type(cfg_item) == dict:
                new_cfg_item = preprocess_config(cfg_item)
                config[cfg_key] = new_cfg_item
        return config
    
    with open(os.path.join('.', 'config', json_filename), 'r') as F:
        config_args = json.load(F)
    config = preprocess_config(config_args)
    return config

def rupdate(orig_dict, update_dict):
    updated_dict = {}
    for key in orig_dict.keys():
        if not key in update_dict.keys():
            updated_dict[key] = orig_dict[key]
    for key in update_dict.keys():
        if not key in orig_dict.keys():
            updated_dict[key] = update_dict[key]
        else:
            if type(update_dict[key]) == dict:
                updated_dict[key] = rupdate(orig_dict[key], update_dict[key])
            else:
                updated_dict[key] = update_dict[key]
    return updated_dict

def get_clargs():
    parser = argparse.ArgumentParser(
        description='Scripts to simulate PowerShield implementations with different configurations and power trace datasets.'
    )
    parser.add_argument(
        'trial_type', metavar='trial-type',
        choices=['hsweep', 'eval'],
        help='What type of trial to run -- a hyperparameter sweep, or an evaluation using a fixed set of hyperparameters.'
    )
    parser.add_argument(
        'protection_method', metavar='protection-method',
        choices=['none', 'randnoise', 'autoencoder', 'gan'],
        help='How the power traces should be protected from the discriminator.'
    )
    parser.add_argument(
        '--config',
        default=None,
        help='Filename of a json file in the \'./config\' folder containing trial parameters. If none is provided, the trial will use default parameters. \'.json\' suffix should be excluded -- e.g. passing \'test\' will tell the trial to use parameter settings specified in \'./config/test.json\'.'
    )
    args = parser.parse_args()
    return args

def get_trial_params(clargs):
    config = {}
    default_config = get_json_args('default_settings.json')
    if 'common' in default_config.keys():
        config = rupdate(config, default_config['common'])
    trial_key = '{}_{}'.format(clargs.trial_type, clargs.protection_method)
    if trial_key in default_config.keys():
        config = rupdate(config, default_config[trial_key])
    if clargs.config is not None:
        override_config = get_json_args(clargs.config+'.json')
        config = rupdate(config, override_config)
    return config

def main():
    clargs = get_clargs()
    trial_params = get_trial_params(clargs)
    if clargs.trial_type == 'eval':
        run_trial(clargs.protection_method, **trial_params)
    elif clargs.trial_type == 'hsweep':
        raise NotImplementedError
    else:
        assert False

if __name__ == '__main__':
    main()