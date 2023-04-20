import random
import numpy as np
import os
import pickle
import torch
import argparse
from train_classifier_on_google_scaaml_dataset import main as run_trial

HPARAM_SAMPLERS = {
    'noise_scale':       lambda: 10**np.random.uniform(-3, 0),
    'weight_decay':      lambda: np.random.choice([0, 10**np.random.uniform(-7, -3)]),
    'max_lr':            lambda: 10**np.random.uniform(-3, -1),
    'pct_start':         lambda: np.random.uniform(0.1, 0.5),
    'dropout':           lambda: np.random.uniform(0, 0.2),
    'target_repr':       lambda: np.random.choice(['bytes', 'bits']),
    'target_attack_pts': lambda: np.random.choice([['sub_bytes_in'], ['sub_bytes_in', 'sub_bytes_out']]),
    'target_bytes':      lambda: np.random.choice([[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]])
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int, help='Random seed to use for this trial.')
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
    save_dir = os.path.join('.', 'results', 'google_scaaml_gridsearch__{}_{}'.format(seed, device))
    os.makedirs(save_dir, exist_ok=True)
    
    trial_idx = 0
    while True:
        hparams = {hparam_name: hparam_sampler() for hparam_name, hparam_sampler in HPARAM_SAMPLERS.items()}
        print('Starting trial {} with seed {} and device {}.'.format(trial_idx, seed, device))
        print('Hyperparameters:')
        print(hparams)
        results, model = run_trial(**hparams, device=device)
        trial_dir = os.path.join(save_dir, 'trial_{}'.format(trial_idx))
        os.makedirs(trial_dir, exist_ok=True)
        torch.save(model, os.path.join(trial_dir, 'best_model.pth'))
        with open(os.path.join(trial_dir, 'results.pickle'), 'wb') as F:
            pickle.dump(results, F)
        trial_idx += 1

if __name__ == '__main__':
    main()