import os
import argparse
import json
import time
import datetime

import utils
import results
import datasets
import models

def printl(s=''):
    utils.printl('(main): ' + s)

def load_experiment(config_path):
    with open(config_path, 'r') as F:
        exp_params = json.load(F)
    return exp_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='.json file listing experiment configuration settings')
    
    args = parser.parse_args()
    config_path = os.path.join(os.getcwd(), 'config', args.config)
    exp_params = load_experiment(config_path)
    
    if 'output_path' in exp_params:
        output_path = exp_params['output_path']
    else:
        dt = datetime.datetime.now()
        output_path = os.path.join(os.getcwd(), 'results',
                                   '%d-%d-%d_%d-%d-%d'%(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
        os.mkdir(output_path)
    if not os.path.isdir(output_path):
        raise Exception('Output path \'{}\' does not point to a directory.'.format(output_path))
    utils.init_printl(output_path)
    printl('Beginning experiments.')
    printl('\tConfig path: \'{}\''.format(config_path))
    printl('\tOutput path: \'{}\''.format(output_path))
    printl()
    
    if 'random_seed' in exp_params:
        random_seed = exp_params['random_seed']
    else:
        random_seed = time.time_ns()%0xFFFFFFFF
    printl('Setting random seed to {}.'.format(random_seed))
    utils.set_random_seed(random_seed)
    printl('\tDone.')
    
    printl('Loading dataset.')
    if not('trace_length' in exp_params['dataset_kwargs']):
        trace_length = 20000
    else:
        trace_length = exp_params['dataset_kwargs']['trace_length']
        del exp_params['dataset_kwargs']['trace_length']
    dataset = datasets.get_dataset(exp_params['dataset'], exp_params['generator_kwargs']['plaintext_encoding'], exp_params['num_keys'], trace_length, **exp_params['dataset_kwargs'])
    printl('\tDone.')
    printl()
    
    printl('Loading generators.')
    valid_keys = dataset.get_valid_keys()
    generators = models.get_generators(valid_keys, exp_params['generator_type'], trace_length, **exp_params['generator_kwargs'])
    printl('\tDone.')
    printl()
    
    printl('Loading discriminator.')
    discriminator = models.get_discriminator(exp_params['discriminator_type'], trace_length, **exp_params['discriminator_kwargs'])
    printl('\tDone.')
    printl()
    
    printl('Loading GAN.')
    gan_kwargs = {
        'gen_optimizer': exp_params['generator_kwargs']['gen_optimizer'],
        'gen_optimizer_kwargs': exp_params['generator_kwargs']['gen_optimizer_kwargs'],
        'gen_loss': exp_params['generator_kwargs']['gen_loss'],
        'gen_loss_kwargs': exp_params['generator_kwargs']['gen_loss_kwargs'],
        'disc_optimizer': exp_params['discriminator_kwargs']['disc_optimizer'],
        'disc_optimizer_kwargs': exp_params['discriminator_kwargs']['disc_optimizer_kwargs'],
        'disc_loss': exp_params['discriminator_kwargs']['disc_loss'],
        'disc_loss_kwargs': exp_params['discriminator_kwargs']['disc_loss_kwargs']}
    gan = models.get_gan(generators, discriminator, **gan_kwargs)
    printl('\tDone.')
    printl()
    
    printl('Training GAN.')
    gan.train(dataset, **exp_params['training_kwargs'])
    printl('\tDone.')
    printl()
    
    printl('Saving models.')
    results.save_model(gan, os.path.join(output_path, 'trained_gan.pth'))
    printl('\tDone.')
    printl()
    
    printl('Saving results.')
    res = results.display_results(gan, **exp_params['display_kwards'])
    results.save_results(res, output)
    printl('\tDone.')
    printl()

if __name__ == '__main__':
    main()