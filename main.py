import pickle
import os
import json
from utils import log_print as print, set_log_file, print_dict
from torch import nn, optim
import generator_models
import discriminator_models
import loss_functions
import results

from multigen_experiment import multigen_experiment

def parse_json(json_filepath):
    with open(json_filepath, 'r') as F:
        config_params = json.load(F)
    
    byte = config_params['byte']
    assert type(byte) == int 
    assert 0 <= byte < 16
    
    keys = config_params['keys']
    if keys == []:
        keys = [x for x in range(256)]
    assert type(keys) == list
    assert all(type(k) == int for k in keys)
    assert all(0 <= k < 256 for k in keys)
    
    key_dataset_kwargs = config_params['key_dataset_kwargs']
    assert type(key_dataset_kwargs) == dict
    
    dataset_prop_for_validation = config_params['dataset_prop_for_validation']
    assert type(dataset_prop_for_validation) == float
    assert 0 <= dataset_prop_for_validation <= 1
    
    dataloader_kwargs = config_params['dataloader_kwargs']
    assert type(dataloader_kwargs) == dict
    
    trace_map_constructor = config_params['trace_map_constructor']
    if trace_map_constructor != 'None':
        trace_map_constructor = getattr(generator_models, trace_map_constructor)
    else:
        trace_map_constructor = None
    
    trace_map_kwargs = config_params['trace_map_kwargs']
    assert type(trace_map_kwargs) == dict
    if 'hidden_activation' in trace_map_kwargs:
        trace_map_kwargs['hidden_activation'] = getattr(nn, trace_map_kwargs['hidden_activation'])
        
    plaintext_map_constructor = config_params['plaintext_map_constructor']
    if plaintext_map_constructor != 'None':
        plaintext_map_constructor = getattr(generator_models, plaintext_map_constructor)
    else:
        plaintext_map_constructor = None
    
    plaintext_map_kwargs = config_params['plaintext_map_kwargs']
    assert type(plaintext_map_kwargs) == dict
    if 'hidden_activation' in plaintext_map_kwargs:
        plaintext_map_kwargs['hidden_activation'] = getattr(nn, plaintext_map_kwargs['hidden_activation'])
        
    key_map_constructor = config_params['key_map_constructor']
    if key_map_constructor != 'None':
        key_map_constructor = getattr(generator_models, key_map_constructor)
    else:
        key_map_constructor = None
    
    key_map_kwargs = config_params['key_map_kwargs']
    assert type(key_map_kwargs) == dict
    if 'hidden_activation' in key_map_kwargs:
        key_map_kwargs['hidden_activation'] = getattr(nn, key_map_kwargs['hidden_activation'])
        
    cumulative_map_constructor = config_params['cumulative_map_constructor']
    if cumulative_map_constructor != 'None':
        cumulative_map_constructor = getattr(generator_models, cumulative_map_constructor)
    else:
        cumulative_map_constructor = None
    
    cumulative_map_kwargs = config_params['cumulative_map_kwargs']
    assert type(cumulative_map_kwargs) == dict
    if 'hidden_activation' in cumulative_map_kwargs:
        cumulative_map_kwargs['hidden_activation'] = getattr(nn, cumulative_map_kwargs['hidden_activation'])
        
    discriminator_constructor = config_params['discriminator_constructor']
    discriminator_constructor = getattr(discriminator_models, discriminator_constructor)
    
    discriminator_kwargs = config_params['discriminator_kwargs']
    assert type(discriminator_kwargs) == dict
    
    try:
        discriminator_loss_constructor = getattr(loss_functions, config_params['discriminator_loss_constructor'])
    except:
        discriminator_loss_constructor = getattr(nn, config_params['discriminator_loss_constructor'])
    
    discriminator_loss_kwargs = config_params['discriminator_loss_kwargs']
    assert type(discriminator_loss_kwargs) == dict
    
    discriminator_optimizer_constructor = getattr(optim, config_params['discriminator_optimizer_constructor'])
    
    discriminator_optimizer_kwargs = config_params['discriminator_optimizer_kwargs']
    assert type(discriminator_optimizer_kwargs) == dict
    
    try:
        generator_loss_constructor = getattr(loss_functions, config_params['generator_loss_constructor'])
    except:
        generator_loss_constructor = getattr(nn, config_params['generator_loss_constructor'])
    
    generator_loss_kwargs = config_params['generator_loss_kwargs']
    assert type(generator_loss_kwargs) == dict
    
    generator_optimizer_constructor = getattr(optim, config_params['generator_optimizer_constructor'])
    
    generator_optimizer_kwargs = config_params['generator_optimizer_kwargs']
    assert type(generator_optimizer_kwargs) == dict
    
    device = config_params['device']
    assert device in ['cpu', 'cuda']
    
    discriminator_pretraining_epochs = config_params['discriminator_pretraining_epochs']
    assert type(discriminator_pretraining_epochs) == int
    assert discriminator_pretraining_epochs >= 0
    
    generator_pretraining_epochs = config_params['generator_pretraining_epochs']
    assert type(generator_pretraining_epochs) == int
    assert generator_pretraining_epochs >= 0
    
    gan_training_epochs = config_params['gan_training_epochs']
    assert type(gan_training_epochs) == int
    assert gan_training_epochs >= 0
    
    discriminator_posttraining_epochs = config_params['discriminator_posttraining_epochs']
    assert type(discriminator_posttraining_epochs) == int
    assert discriminator_posttraining_epochs >= 0
    
    seed = config_params['seed']
    assert type(seed) == int
    
    special_evaluation_methods = config_params['special_evaluation_methods']
    assert type(special_evaluation_methods) == list
    assert all([type(m) == str for m in special_evaluation_methods])
    for (idx, m) in enumerate(special_evaluation_methods):
        special_evaluation_methods[idx] = getattr(results, m)
    
    special_evaluation_methods_period = config_params['special_evaluation_methods_period']
    assert type(special_evaluation_methods_period) == int
    assert special_evaluation_methods_period >= 0
    
    exp_type = config_params['exp_type']
    
    exp_kwargs = {'byte': byte,
                  'keys': keys,
                  'key_dataset_kwargs': key_dataset_kwargs,
                  'dataloader_kwargs': dataloader_kwargs,
                  'dataset_prop_for_validation': dataset_prop_for_validation,
                  'trace_map_constructor': trace_map_constructor,
                  'trace_map_kwargs': trace_map_kwargs,
                  'plaintext_map_constructor': plaintext_map_constructor,
                  'plaintext_map_kwargs': plaintext_map_kwargs,
                  'key_map_constructor': key_map_constructor,
                  'key_map_kwargs': key_map_kwargs,
                  'cumulative_map_constructor': cumulative_map_constructor,
                  'cumulative_map_kwargs': cumulative_map_kwargs,
                  'discriminator_constructor': discriminator_constructor,
                  'discriminator_kwargs': discriminator_kwargs,
                  'discriminator_loss_constructor': discriminator_loss_constructor,
                  'discriminator_loss_kwargs': discriminator_loss_kwargs,
                  'discriminator_optimizer_constructor': discriminator_optimizer_constructor,
                  'discriminator_optimizer_kwargs': discriminator_optimizer_kwargs,
                  'generator_loss_constructor': generator_loss_constructor,
                  'generator_loss_kwargs': generator_loss_kwargs,
                  'generator_optimizer_constructor': generator_optimizer_constructor,
                  'generator_optimizer_kwargs': generator_optimizer_kwargs,
                  'device': device,
                  'discriminator_pretraining_epochs': discriminator_pretraining_epochs,
                  'generator_pretraining_epochs': generator_pretraining_epochs,
                  'gan_training_epochs': gan_training_epochs,
                  'discriminator_posttraining_epochs': discriminator_posttraining_epochs,
                  'seed': seed,
                  'special_evaluation_methods': special_evaluation_methods,
                  'special_evaluation_methods_period': special_evaluation_methods_period}
    
    return exp_type, exp_kwargs

def main():
    output_path = os.path.join('.', 'results')
    config_path = os.path.join('.', 'config')
    config_files = [f for f in os.listdir(config_path) if '.json' in f]
    
    for config_file in config_files:
        output_dir = config_file[:-5]
        if not(os.path.exists(os.path.join(output_path, output_dir))):
            os.mkdir(os.path.join(output_path, output_dir))
        set_log_file(os.path.join(output_path, output_dir, 'output.txt'))
        print('Beginning trial described in {}.'.format(os.path.join(config_path, config_file)))
        
        exp_type, exp_kwargs = parse_json(os.path.join(config_path, config_file))
        if exp_type == 'multigen':
            print('Experiment type: multiple generators each corresponding to 1 key.')
            print('Experiment settings:')
            print_dict(exp_kwargs, prefix='\t')
            results = multigen_experiment(**exp_kwargs)
        else:
            raise ValueError('Invalid exp_type: \'{}\''.format(exp_type))
        
        with open(os.path.join(output_path, output_dir, 'results.pickle'), 'wb') as F:
            pickle.dump(results, F)

if __name__ == '__main__':
    main()