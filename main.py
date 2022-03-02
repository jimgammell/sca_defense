import os
import argparse
import json
import time
import datetime

import utils
import results
import datasets
import models

def printl(s: str = ''):
    utils.printl('(main):'.ljust(utils.get_pad_width()) + s)

def load_experiment(config_path: str) -> dict:    
    if config_path != None:
        if not os.path.exists(config_path):
            raise ValueError('No file found at specified configuration path \'{}\''.format(config_path))
        with open(config_path, 'r') as F:
            config_params = json.load(F)
    else:
        config_params = {}
    
    if 'output_path' in config_params:
        output_path = config_params['output_path']
    else:
        dt = datetime.datetime.now()
        output_path = os.path.join(os.getcwd(), 'results',
          '%d-%d-%d_%d-%d-%d'%(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second))
    results_path = os.path.join(os.getcwd(), 'results')
    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    if os.path.isdir(output_path):
        raise Exception('output_path \'{}\' already exists'.format(output_path))
    else:
        os.mkdir(output_path)
    if not os.path.isdir(output_path):
        raise ValueError('output_path \'{}\' does not point to a directory.'.format(output_path))
    utils.init_printl(output_path)
    
    # Delaying initial print statements until location of log file has been determined
    printl('Parsing configuration settings...')
    if config_path != None:
        printl('\tConfiguration file path: \'{}\''.format(config_path))
    else:
        printl('\tNo configuration file path provided. Using default settings.')
    if 'output_path' in config_params:
        del config_params['output_path']
        printl('\tUsing specified output path: \'{}\''.format(output_path))
    else:
        printl('\tUsing default output path: \'{}\''.format(output_path))
        
    if 'random_seed' in config_params:
        random_seed = config_params['random_seed']
        del config_params['random_seed']
        printl('\tUsing specified random seed: {}'.format(random_seed))
    else:
        random_seed = time.time_ns()%0xFFFFFFFF
        printl('\tUsing default random seed: {}'.format(random_seed))
    if not(type(random_seed) == int):
        raise TypeError('random_seed must be of type {} but is of type {}'.format(int, type(random_seed)))
    
    if 'num_keys' in config_params:
        num_keys = config_params['num_keys']
        del config_params['num_keys']
        printl('\tUsing specified number of keys: {}'.format(num_keys))
    else:
        num_keys = 16
        printl('\tUsing default number of keys: {}'.format(num_keys))
    if not(type(num_keys) == int):
        raise TypeError('num_keys must be of type {} but is of type {}'.format(int, type(num_keys)))
    
    if 'trace_length' in config_params:
        trace_length = config_params['trace_length']
        del config_params['trace_length']
        printl('\tUsing specified trace length: {}'.format(trace_length))
    else:
        trace_length = 20000
        printl('\tUsing default trace length: {}'.format(trace_length))
    if not(type(trace_length) == int):
        raise TypeError('trace_length must be of type {} but is of type {}'.format(int, type(trace_length)))
    
    if 'attack_point' in config_params:
        attack_point = config_params['attack_point']
        del config_params['attack_point']
        printl('\tUsing specified attack point: {}'.format(attack_point))
    else:
        attack_point = 'sub_bytes_in'
        printl('\tUsing default attack point: {}'.format(attack_point))
    if not(type(attack_point) == str):
        raise TypeError('attack_point must be of type {} but is of type {}'.format(str, type(attack_point)))
    
    if 'byte' in config_params:
        byte = config_params['byte']
        del config_params['byte']
        printl('\tUsing specified byte: {}'.format(byte))
    else:
        byte = 0
        printl('\tUsing default byte: {}'.format(byte))
    if not(type(byte) == int):
        raise TypeError('byte must be of type {} but is of type {}'.format(int, type(byte)))
    
    if 'dataset' in config_params:
        dataset = config_params['dataset']
        del config_params['dataset']
        printl('\tUsing specified dataset: {}'.format(dataset))
    else:
        dataset = 'google_scaaml'
        printl('\tUsing default dataset: {}'.format(dataset))
    if not(type(dataset) == str):
        raise TypeError('dataset must be of type {} but is of type {}'.format(str, type(dataset)))
    
    if 'dataset_kwargs' in config_params:
        dataset_kwargs = config_params['dataset_kwargs']
        del config_params['dataset_kwargs']
        printl('\tUsing specified dataset kwargs: {}'.format(dataset_kwargs))
    else:
        dataset_kwargs = {}
        printl('\tUsing no dataset kwargs')
    if not(type(dataset_kwargs) == dict):
        raise TypeError('dataset_kwargs must be of type {} but is of type {}'.format(dict, type(dataset_kwargs)))
        
    if 'generator_type' in config_params:
        generator_type = config_params['generator_type']
        del config_params['generator_type']
        printl('\tUsing specified generator type: {}'.format(generator_type))
    else:
        generator_type = 'mlp'
        printl('\tUsing default generator type: {}'.format(generator_type))
    if not(type(generator_type) == str):
        raise TypeError('generator_type must be of type {} but is of type {}'.format(str, type(generator_type)))
        
    if 'generator_kwargs' in config_params:
        generator_kwargs = config_params['generator_kwargs']
        del config_params['generator_kwargs']
        printl('\tUsing specified generator kwargs: {}'.format(generator_kwargs))
    else:
        generator_kwargs = {}
        printl('\tUsing no generator kwargs')
    if not(type(generator_kwargs) == dict):
        raise TypeError('generator_kwargs must be of type {} but is of type {}'.format(dict, type(generator_kwargs)))
        
    if 'generator_plaintext_encoding' in config_params:
        generator_plaintext_encoding = config_params['generator_plaintext_encoding']
        del config_params['generator_plaintext_encoding']
        printl('\tUsing specified generator plaintext encoding: {}'.format(generator_plaintext_encoding))
    else:
        generator_plaintext_encoding = 'binary'
        printl('\tUsing default generator plaintext encoding: {}'.format(generator_plaintext_encoding))
    if not(type(generator_plaintext_encoding) == str):
        raise TypeError('generator_plaintext_encoding must be of type {} but is of type {}'.format(str, type(generator_plaintext_encoding)))
    
    if 'generator_batch_size' in config_params:
        generator_batch_size = config_params['generator_batch_size']
        del config_params['generator_batch_size']
        printl('\tUsing specified generator batch size: {}'.format(generator_batch_size))
    else:
        generator_batch_size = 16
        printl('\tUsing default generator batch size: {}'.format(generator_batch_size))
    if not(type(generator_batch_size) == int):
        raise TypeError('generator_batch_size must be of type {} but is of type {}'.format(int, type(generator_batch_size)))
        
    if 'generator_optimizer' in config_params:
        generator_optimizer = config_params['generator_optimizer']
        del config_params['generator_optimizer']
        printl('\tUsing specified generator optimizer: {}'.format(generator_optimizer))
    else:
        generator_optimizer = 'SGD'
        printl('\tUsing default generator optimizer: {}'.format(generator_optimizer))
    if not(type(generator_optimizer) == str):
        raise TypeError('generator_optimizer must be of type {} but is of type {}'.format(str, type(generator_optimizer)))
        
    if 'generator_optimizer_kwargs' in config_params:
        generator_optimizer_kwargs = config_params['generator_optimizer_kwargs']
        del config_params['generator_optimizer_kwargs']
        printl('\tUsing specified generator kwargs: {}'.format(generator_optimizer_kwargs))
    else:
        generator_optimizer_kwargs = {}
        printl('\tUsing no generator kwargs')
    if not(type(generator_optimizer_kwargs) == dict):
        raise TypeError('generator_optimizer_kwargs must be of type {} but is of type {}'.format(dict, type(generator_optimizer_kwargs)))
        
    if 'generator_loss' in config_params:
        generator_loss = config_params['generator_loss']
        del config_params['generator_loss']
        printl('\tUsing specified generator loss: {}'.format(generator_loss))
    else:
        generator_loss = 'CategoricalCrossentropy'
        printl('\tUsing default generator loss: {}'.format(generator_loss))
    if not(type(generator_loss) == str):
        raise TypeError('generator_loss must be of type {} but is of type {}'.format(str, type(generator_loss)))
        
    if 'generator_loss_kwargs' in config_params:
        generator_loss_kwargs = config_params['generator_loss_kwargs']
        printl('\tUsing specified generator loss kwargs: {}'.format(generator_loss_kwargs))
    else:
        generator_loss_kwargs = {}
        printl('\tUsing no generator loss kwargs')
    if not(type(generator_loss_kwargs) == dict):
        raise TypeError('generator_loss_kwargs must be of type {} but is of type {}'.format(dict, type(generator_loss_kwargs)))
        
    if 'discriminator_type' in config_params:
        discriminator_type = config_params['discriminator_type']
        del config_params['discriminator_type']
        printl('\tUsing specified discriminator type: {}'.format(discriminator_type))
    else:
        discriminator_type = 'google_resnet1d'
        printl('\tUsing default discriminator type: {}'.format(discriminator_type))
    if not(type(discriminator_type) == str):
        raise TypeError('discriminator_type must be of type {} but is of type {}'.format(str, type(discriminator_type)))
        
    if 'discriminator_optimizer' in config_params:
        discriminator_optimizer = config_params['discriminator_optimizer']
        del config_params['discriminator_optimizer']
        printl('\tUsing specified discriminator optimizer: {}'.format(discriminator_optimizer))
    else:
        discriminator_optimizer = 'SGD'
        printl('\tUsing default discriminator optimizer: {}'.format(discriminator_optimizer))
    if not(type(discriminator_optimizer) == str):
        raise TypeError('discriminator_optimizer must be of type {} but is of type {}'.format(str, type(discriminator_optimizer)))
        
    if 'discriminator_optimizer_kwargs' in config_params:
        discriminator_optimizer_kwargs = config_params['discriminator_optimizer_kwargs']
        del config_params['discriminator_optimizer_kwargs']
        printl('\tUsing specified discriminator optimizer kwargs: {}'.format(discriminator_optimizer_kwargs))
    else:
        discriminator_optimizer_kwargs = {}
        printl('\tUsing no discriminator optimizer kwargs')
    if not(type(discriminator_optimizer_kwargs) == dict):
        raise TypeError('discriminator_optimizer_kwargs must be of type {} but is of type {}'.format(dict, type(discriminator_optimizer_kwargs)))
        
    if 'discriminator_loss' in config_params:
        discriminator_loss = config_params['discriminator_loss']
        del config_params['discriminator_loss']
        printl('\tUsing specified discriminator loss: {}'.format(discriminator_loss))
    else:
        discriminator_loss = 'CategoricalCrossentropy'
        printl('\tUsing default discriminator loss: {}'.format(discriminator_loss))
    if not(type(discriminator_loss) == str):
        raise TypeError('discriminator_loss must be of type {} but is of type {}'.format(str, type(discriminator_loss)))
        
    if 'discriminator_loss_kwargs' in config_params:
        discriminator_loss_kwargs = config_params['discriminator_loss_kwargs']
        del config_params['discriminator_loss_kwargs']
        printl('\tUsing specified discriminator loss kwargs: {}'.format(discriminator_loss_kwargs))
    else:
        discriminator_loss_kwargs = {}
        printl('\tUsing no discriminator loss kwargs')
    if not(type(discriminator_loss_kwargs) == dict):
        raise TypeError('discriminator_loss_kwargs must be of type {} but is of type {}'.format(dict, type(discriminator_loss_kwargs)))
        
    if 'training_kwargs' in config_params:
        training_kwargs = config_params['training_kwargs']
        del config_params['training_kwargs']
        printl('\tUsing specified training kwargs: {}'.format(training_kwargs))
    else:
        training_kwargs = {}
        printl('\tUsing no training kwargs')
    if not(type(training_kwargs) == dict):
        raise TypeError('training_kwargs must be of type {} but is of type {}'.format(dict, type(training_kwargs)))
        
    if 'display_kwargs' in config_params:
        display_kwargs = config_params['display_kwargs']
        del config_params['display_kwargs']
        printl('\tUsing specified display kwargs: {}'.format(display_kwargs))
    else:
        display_kwargs = {}
        printl('\tUsing no display kwargs')
    if not(type(display_kwargs) == dict):
        raise TypeError('display_kwargs must be of type {} but is of type {}'.format(dict, type(display_kwargs)))
        
    printl('Done.')
    if len(config_params) != 0:
        printl('Warning: the following parameters in config file are unrecognized.')
        for key in config_params:
            printl('\t{}: {}'.format(key, config_params[key]))
    printl()
    
    exp_params = {
        'output_path': output_path,
        'random_seed': random_seed,
        'num_keys': num_keys,
        'trace_length': trace_length,
        'attack_point': attack_point,
        'byte': byte,
        'dataset': dataset,
        'dataset_kwargs': dataset_kwargs,
        'generator_type': generator_type,
        'generator_kwargs': generator_kwargs,
        'generator_plaintext_encoding': generator_plaintext_encoding,
        'generator_batch_size': generator_batch_size,
        'generator_optimizer': generator_optimizer,
        'generator_optimizer_kwargs': generator_optimizer_kwargs,
        'generator_loss': generator_loss,
        'generator_loss_kwargs': generator_loss_kwargs,
        'discriminator_type': discriminator_type,
        'discriminator_optimizer': discriminator_optimizer,
        'discriminator_optimizer_kwargs': discriminator_optimizer_kwargs,
        'discriminator_loss': discriminator_loss,
        'discriminator_loss_kwargs': discriminator_loss_kwargs,
        'training_kwargs': training_kwargs,
        'display_kwargs': display_kwargs }
    
    return exp_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, action='store', default=None, dest='config', 
                       help='.json file listing experiment configuration settings')
    parser.add_argument('--display', '-d', action='store_true', default=False, dest='display',
                       help='Generate plots to visualize GAN training')
    
    args = parser.parse_args()
    if args.config != None:
        config_path = os.path.join(os.getcwd(), 'config', args.config)
    else:
        config_path = None
    exp_params = load_experiment(config_path)
    
    printl('Beginning experiments.')
    printl('\tOutput path: \'{}\''.format(exp_params['output_path']))
    printl()
    
    printl('Setting random seed to {}.'.format(exp_params['random_seed']))
    utils.set_random_seed(exp_params['random_seed'])
    printl('Done.')
    printl()
    
    printl('Loading dataset.')
    dataset_kwargs = {
        'name': exp_params['dataset'],
        'plaintext_encoding': exp_params['generator_plaintext_encoding'],
        'num_keys': exp_params['num_keys'],
        'trace_length': exp_params['trace_length'],
        'attack_point': exp_params['attack_point'],
        'byte': exp_params['byte'],
        'generator_batch_size': exp_params['generator_batch_size']}
    dataset_kwargs.update(exp_params['dataset_kwargs'])
    dataset = datasets.get_dataset(**dataset_kwargs)
    printl('Done.')
    printl()
    
    printl('Loading generators.')
    valid_keys = dataset.get_valid_keys()
    generator_kwargs = {
        'keys': valid_keys,
        'gen_type': exp_params['generator_type'],
        'trace_length': exp_params['trace_length']}
    generator_kwargs.update(exp_params['generator_kwargs'])
    generators = models.get_generators(**generator_kwargs)
    printl('Done.')
    printl()
    
    printl('Loading discriminator.')
    discriminator = models.get_discriminator(exp_params['discriminator_type'], trace_length, **exp_params['discriminator_kwargs'])
    printl('\tDone.')
    printl()
    
    assert False
    
    printl('Loading GAN.')
    gan_kwargs = {
        'generators': generators,
        'discriminator': discriminator,
        'gen_optimizer': exp_params['generator_optimizer'],
        'gen_optimizer_kwargs': exp_params['generator_optimizer_kwargs'],
        'gen_loss': exp_params['generator_loss'],
        'gen_loss_kwargs': exp_params['generator_loss_kwargs'],
        'disc_optimizer': exp_params['discriminator_optimizer'],
        'disc_optimizer_kwargs': exp_params['discriminator_optimizer_kwargs'],
        'disc_loss': exp_params['discriminator_loss'],
        'disc_loss_kwargs': exp_params['discriminator_loss_kwargs']}
    gan = models.get_gan(**gan_kwargs)
    printl('Done.')
    printl()
    
    printl('Training GAN.')
    training_kwargs = {
        'dataset': dataset}
    training_kwargs.update(exp_params['training_kwargs'])
    training_results = gan.train(**training_kwargs)
    printl('Done.')
    printl()
    
    printl('Saving models.')
    save_model_kwargs = {
        'dest': exp_params['output_path']}
    gan.save(**save_model_kwargs)
    printl('Done.')
    printl()
    
    printl('Saving results.')
    if args.display:
        display_kwargs = {
            'results': training_results}
        display_kwargs.update(exp_params['display_kwargs'])
        figures = results.display_results(**display_kwargs)
    else:
        figures = []
    save_results_kwargs = {
        'figures': figures,
        'results': training_results,
        'dest': exp_params['output']}
    results.save_results(**save_results_kwargs)
    printl('Done.')
    printl()

if __name__ == '__main__':
    main()