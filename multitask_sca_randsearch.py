import random
import numpy as np
import torch
import os
import pickle
import argparse
from multitask_sca_trial import run_trial

HPARAM_SAMPLERS = {
    'disc_steps_per_gen_step': lambda: np.random.choice((1.0, 2.0, 3.0)),
    'l1_rec_coefficient': lambda: 10**np.random.uniform(-5, 0),
    'gen_classification_coefficient': lambda: np.random.choice((2**np.random.uniform(-3, -1), 1-2**np.random.uniform(-3, -1))),
    'average_deviation_penalty': lambda: np.random.choice((0.0, 10**np.random.uniform(-2, 0)))
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int, help='Random seed for this trial.')
    parser.add_argument('--device', default=None, type=str, help='Device on which to run this trial.')
    args = parser.parse_args()
    
    if args.seed is None:
        seed = 0
    else:
        seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = os.path.join('.', 'results', 'sca_cyclegan_randsearch__{}_{}'.format(seed, device))
    os.makedirs(save_dir, exist_ok=True)
    
    trial_idx = 0
    while True:
        hparams = {hparam_name: hparam_sampler() for hparam_name, hparam_sampler in HPARAM_SAMPLERS.items()}
        print('Starting trial {} with seed {} and device {}.'.format(trial_idx, seed, device))
        print('Hyperparameters:')
        print(hparams)
        run_trial(save_dir=os.path.join(save_dir, 'trial_{}'.format(trial_idx)), trial_info=hparams, **hparams)
        print('\n\n')

if __name__ == '__main__':
    main()