import os
import argparse

def get_clargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'trial-type',
        choices=['eval', 'tune'],
        help='Whether to do hyperparameter tuning with Ray Tune, or to evaluate network performance with a given set of hyperparameters.'
    )
    parser.add_argument(
        'protection-type',
        choices=['unprotected', 'randnoise', 'autoencoder', 'adversarial'],
        help='What form of protection to use on the power traces.'
    )
    parser.add_argument(
        '--config',
        help='Path of json file specifying arguments to the trial.'
    )
    args = parser.parse_args()
    return args

def main():
    clargs = get_clargs()

if __name__ == '__main__':
    main()