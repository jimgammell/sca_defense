from torch.utils.data import random_split, DataLoader
from torchvision import transforms

import random
from dataset import AesSingleKeyDataset, AesKeyGroupDataset, IntToOnehot, IntToBinary, ToTensor1D, NormTensorMagnitude
import torch
import generator_models
import discriminator_models
from utils import log_print as print
from tqdm import tqdm
import numpy as np
from train import *
from results import Results

def _construct_generator(dataset, keys,
                         trace_map_constructor, trace_map_kwargs,
                         plaintext_map_constructor, plaintext_map_kwargs,
                         key_map_constructor, key_map_kwargs,
                         cumulative_map_constructor, cumulative_map_kwargs):
    generators = {}
    for key in keys:
        trace_map = trace_map_constructor(dataset.trace_size, dataset.trace_size, **trace_map_kwargs)
        plaintext_map = plaintext_map_constructor(dataset.plaintext_size, dataset.plaintext_size, **plaintext_map_kwargs)
        key_map = key_map_constructor(dataset.key_size, dataset.key_size, **key_map_kwargs)
        cumulative_map = cumulative_map_constructor(np.sum([np.prod(list(s)) for s in [dataset.key_size, dataset.trace_size, dataset.plaintext_size]]), dataset.trace_size, **cumulative_map_kwargs)
        generator = generator_models.Generator(trace_map, plaintext_map, key_map, cumulative_map)
        generators.update({key: generator})
    composite_generator = generator_models.CompositeGenerator(generators)
    return composite_generator

def _construct_discriminator(dataset,
                             discriminator_constructor, discriminator_kwargs):
    discriminator = discriminator_constructor(dataset.trace_size, **discriminator_kwargs)
    discriminator = discriminator_models.Discriminator(discriminator)
    return discriminator

def multigen_experiment(byte,
                        keys,
                        key_dataset_kwargs,
                        dataset_prop_for_validation,
                        dataloader_kwargs,
                        trace_map_constructor,
                        trace_map_kwargs,
                        plaintext_map_constructor,
                        plaintext_map_kwargs,
                        key_map_constructor,
                        key_map_kwargs,
                        cumulative_map_constructor,
                        cumulative_map_kwargs,
                        discriminator_constructor,
                        discriminator_kwargs,
                        discriminator_loss_constructor,
                        discriminator_loss_kwargs,
                        discriminator_optimizer_constructor,
                        discriminator_optimizer_kwargs,
                        generator_loss_constructor,
                        generator_loss_kwargs,
                        generator_optimizer_constructor,
                        generator_optimizer_kwargs,
                        device,
                        discriminator_pretraining_epochs,
                        generator_pretraining_epochs,
                        gan_training_epochs,
                        discriminator_posttraining_epochs,
                        seed,
                        special_evaluation_methods,
                        special_evaluation_methods_period):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create dataset and partition into training and validation sets
    print('Loading datasets.')
    trace_transform = transforms.Compose([ToTensor1D(),
                                          NormTensorMagnitude(1, -1)])
    plaintext_transform = transforms.Compose([IntToBinary(8),
                                              ToTensor1D()])
                                              #ToTensor1D(),
                                              #NormTensorMagnitude(1, -1)])
    key_transform = transforms.Compose([IntToBinary(8),
                                        ToTensor1D()])
                                        #ToTensor1D(),
                                        #NormTensorMagnitude(1, -1)])
    key_datasets = [AesSingleKeyDataset(byte,
                                        key,
                                        trace_transform=trace_transform,
                                        plaintext_transform=plaintext_transform,
                                        **key_dataset_kwargs) for key in keys]
    dataset = AesKeyGroupDataset(key_datasets, byte, key_transform=key_transform)
    print(dataset)
    validation_dataset_length = int(dataset_prop_for_validation*len(dataset))
    training_dataset_length = len(dataset)-validation_dataset_length
    training_dataset, validation_dataset = random_split(dataset, [training_dataset_length, validation_dataset_length])
    training_dataloader = DataLoader(training_dataset, drop_last=True, **dataloader_kwargs)
    validation_dataloader = DataLoader(validation_dataset, drop_last=True, **dataloader_kwargs)
    
    #for _ in range(10):
    #    batch = next(iter(training_dataloader))
    #    key_idx, trace, plaintext, key = batch
    #    print('Key idx shape:', key_idx.size())
    #    print('Trace shape:', trace.size())
    #    print('Plaintext shape:', plaintext.size())
    #    print('Key shape:', key.size())
    #    print('Key idx:', key_idx)
    #    print('Trace max/min:', torch.max(trace), torch.min(trace))
    #    print('Plaintext:', plaintext)
    #    print('Key:', key)
    
    # Create generator and related objects
    print('Constructing generator.')
    if cumulative_map_constructor == None:
        potential_map_constructors = [(trace_map_constructor, trace_map_kwargs), (plaintext_map_constructor, plaintext_map_kwargs), (key_map_constructor, key_map_kwargs)]
        map_constructor = [pc for pc in potential_map_constructors if pc[0] != None]
        assert len(map_constructor) == 1
        map_constructor, map_kwargs = map_constructor[0]
        map = map_constructor(dataset.key_size, dataset.trace_size, **map_kwargs)
        generator = generator_models.KeyOnlyGenerator(map)
    else:
        generator = _construct_generator(dataset, keys, trace_map_constructor, trace_map_kwargs, plaintext_map_constructor, plaintext_map_kwargs, key_map_constructor, key_map_kwargs, cumulative_map_constructor, cumulative_map_kwargs)
    print(generator)
    generator = generator.to(device)
    generator_optimizer = generator_optimizer_constructor(filter(lambda p: p.requires_grad==True, generator.parameters()), **generator_optimizer_kwargs)
    generator_loss = generator_loss_constructor(**generator_loss_kwargs)
    print()
    
    # Create discriminator and related objects
    print('Constructing discriminator.')
    discriminator = _construct_discriminator(dataset, discriminator_constructor, discriminator_kwargs)
    print(discriminator)
    discriminator = discriminator.to(device)
    discriminator_optimizer = discriminator_optimizer_constructor(filter(lambda p: p.requires_grad==True, discriminator.parameters()), **discriminator_optimizer_kwargs)
    discriminator_loss = discriminator_loss_constructor(**discriminator_loss_kwargs)
    print()
    
    test_results()
    results = Results()
    def update_results(results, training_results=None, validation_results=None):
        if training_results != None:
            print('Training results:')
            print(training_results)
            training_results.rename('gen_loss', 'gen_train_loss')
            training_results.rename('disc_loss', 'disc_train_loss')
            training_results.rename('disc_acc', 'disc_train_acc')
            results.extend(training_results)
        if validation_results != None:
            print('Validation results:')
            print(validation_results)
            validation_results.rename('gen_loss', 'gen_val_loss')
            validation_results.rename('disc_loss', 'disc_val_loss')
            validation_results.rename('disc_acc', 'disc_val_acc')
            results.extend(validation_results)
    
    def run_special_evaluations(epoch):
        if special_evaluation_methods_period != 0:
            if epoch % special_evaluation_methods_period == 0:
                se_results = run_special_evaluation_methods(special_evaluation_methods, generator, discriminator, validation_dataloader, device)
                results.extend(se_results)
    
    # Initial results
    print('Calculating initial results.')
    training_results = eval_gan_epoch(training_dataloader, generator, discriminator, generator_loss, discriminator_loss, device)
    validation_results = eval_gan_epoch(validation_dataloader, generator, discriminator, generator_loss, discriminator_loss, device)
    run_special_evaluations(0)
    update_results(results, training_results, validation_results)
    print()
    
    # GAN training
    print('Training discriminator and generator simultaneously.')
    for epoch in range(gan_training_epochs):
        print('\tEpoch', epoch+1)
        training_results = train_gan_epoch(training_dataloader, generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer, device)
        validation_results = eval_gan_epoch(validation_dataloader, generator, discriminator, generator_loss, discriminator_loss, device)
        run_special_evaluations(epoch)
        update_results(results, training_results, validation_results)
        print()
    print()
    
    # Retrain discriminator from scratch to evaluate GAN performance
    print('Training new discriminator on static trained discriminator.')
    discriminator = _construct_discriminator(dataset, discriminator_constructor, discriminator_kwargs)
    discriminator = discriminator.to(device)
    discriminator_optimizer = discriminator_optimizer_constructor(filter(lambda p: p.requires_grad==True, discriminator.parameters()), **discriminator_optimizer_kwargs)
    
    print('\tInitial performance')
    training_results = eval_gan_epoch(training_dataloader, generator, discriminator, generator_loss, discriminator_loss, device)
    validation_results = eval_gan_epoch(validation_dataloader, generator, discriminator, generator_loss, discriminator_loss, device)
    run_special_evaluations(0)
    update_results(results, training_results, validation_results)
    print()
    
    for epoch in range(discriminator_posttraining_epochs):
        print('\tEpoch', epoch+1)
        training_results = train_discriminator_alone_epoch(training_dataloader, generator, discriminator, generator_loss, discriminator_loss, discriminator_optimizer, device)
        validation_results = eval_gan_epoch(validation_dataloader, generator, discriminator, generator_loss, discriminator_loss, device)
        run_special_evaluations(epoch)
        update_results(results, training_results, validation_results)
        print()
    print()
    print(results)
    
    return results.data
    