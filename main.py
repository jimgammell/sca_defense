import os
import argparse
import json

def get_clargs():
    parser = argparse.ArgumentParser(
        description='Scripts to simulate PowerShield implementations with different configurations and power trace datasets.'
    )
    parser.add_argument(
        'trial-type',
        choices=['hsweep', 'eval'],
        help='What type of trial to run -- a hyperparameter sweep, or an evaluation using a fixed set of hyperparameters.'
    )
    parser.add_argument(
        'protection-method',
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
    default_filename = 'default_' + clargs.trial_type + clargs.protection_method + '.json'
    with open(os.path.join('.', 'config', default_filename), 'r') as F:
        trial_params = json.load(F)
    if clargs.config is not None:
        with open(os.path.join('.', 'config', clargs.config), 'r') as F:
            ow_trial_params = json.load(F)
    trial_params.update(ow_trial_params)
    return trial_params

def main():
    clargs = get_clargs()

if __name__ == '__main__':
    main()