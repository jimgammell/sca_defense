import pickle
import os

from multigen_experiment import multigen_experiment

def main():
    config_path = os.path.join('.', 'config')
    config_files = [f for f in os.listdir(config_path) if '.json' in f]
    
    for config_file in config_files:
        with open(os.path.join(config_path, config_file), 'r') as F:
            config_params = json.load(F)
        
        byte = config_params['byte']
        keys = config_params['keys']
        key_dataset_kwargs = config_params['key_dataset_kwargs']
        dataset_prop_for_validation = config_params['dataset_prop_for_validation']
        trace_map_architecture = config_params['trace_map_architecture']
        trace_map_kwargs = config_params['trace_map_kwargs']
        plaintext_map_architecture = config_params['plaintext_map_architecture']
        plaintext_map_kwargs = config_params['plaintext_map_kwargs']
        key_map_architecture = config_params['key_map_architecture']
        key_map_kwargs = config_params['key_map_kwargs']
        cumulative_map_architecture = config_params['cumulative_map_architecture']
        cumulative_map_kwargs = config_params['cumulative_map_kwargs']
        discriminator_architecture = config_params['discriminator_architecture']
        discriminator_kwargs = config_params['discriminator_kwargs']
        discriminator_loss = config_params['discriminator_loss']
        discriminator_loss_kwargs = config_params['discriminator_loss_kwargs']
        discriminator_optimizer = config_params['discriminator_optimizer']
        discriminator_optimizer_kwargs = config_params['discriminator_optimizer_kwargs']
        generator_loss = config_params['generator_loss']
        generator_loss_kwargs = config_params['generator_loss_kwargs']
        generator_optimizer = config_params['generator_optimizer']
        generator_optimizer_kwargs = config_params['generator_optimizer_kwargs']
        
        if exp_type == 'multigen':
            exp_kwargs = {'byte': byte,
                          'keys': keys,
                          'key_dataset_kwargs': key_dataset_kwargs,
                          'dataset_prop_for_validation': dataset_prop_for_validation,
                          'trace_map_architecture': trace_map_architecture,
                          'trace_map_kwargs': trace_map_kwargs,
                          'plaintext_map_architecture': plaintext_map_architecture,
                          'plaintext_map_kwargs': plaintext_map_kwargs,
                          'key_map_architecture': key_map_architecture,
                          'key_map_kwargs': key_map_kwargs,
                          'cumulative_map_architecture': cumulative_map_architecture,
                          'cumulative_map_kwargs': cumulative_map_kwargs,
                          'discriminator_architecture': discriminator_architecture,
                          'discriminator_kwargs': discriminator_kwargs,
                          'discriminator_loss': discriminator_loss,
                          'discriminator_loss_kwargs': discriminator_loss_kwargs,
                          'discriminator_optimizer': discriminator_optimizer,
                          'discriminator_optimizer_kwargs': discriminator_optimizer_kwargs,
                          'generator_loss': generator_loss,
                          'generator_loss_kwargs': generator_loss_kwargs,
                          'generator_optimizer': generator_optimizer,
                          'generator_optimizer_kwargs': generator_optimizer_kwargs}
            results = multigen_experiment(**exp_kwargs)
        else:
            raise ValueError('Invalid exp_type: \'{}\''.format(exp_type))
        
        with open(os.path.join(results_path, results_dir, 'results.pickle'), 'wb') as F:
            pickle.dump(results)

if __name__ == '__main__':
    main()