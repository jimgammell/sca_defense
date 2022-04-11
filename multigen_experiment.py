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
                        seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create dataset and partition into training and validation sets
    print('Loading datasets.')
    trace_transform = transforms.Compose([ToTensor1D(),
                                          NormTensorMagnitude(1, -1)])
    plaintext_transform = transforms.Compose([IntToBinary(8),
                                              ToTensor1D()])
    key_transform = transforms.Compose([IntToBinary(8),
                                        ToTensor1D()])
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
    
    # Create generator and related objects
    print('Constructing generator.')
    generator = _construct_generator(dataset, keys, trace_map_constructor, trace_map_kwargs, plaintext_map_constructor, plaintext_map_kwargs, key_map_constructor, key_map_kwargs, cumulative_map_constructor, cumulative_map_kwargs)
    print(generator)
    generator = generator.to(device)
    generator_optimizer = generator_optimizer_constructor(generator.parameters(), **generator_optimizer_kwargs)
    generator_loss = generator_loss_constructor(**generator_loss_kwargs)
    generator_loss.reduction = 'none'
    print()
    
    # Create discriminator and related objects
    print('Constructing discriminator.')
    discriminator = _construct_discriminator(dataset, discriminator_constructor, discriminator_kwargs)
    print(discriminator)
    discriminator = discriminator.to(device)
    discriminator_optimizer = discriminator_optimizer_constructor(discriminator.parameters(), **discriminator_optimizer_kwargs)
    discriminator_loss = discriminator_loss_constructor(**discriminator_loss_kwargs)
    discriminator_loss.reduction = 'none'
    print()
    
    def print_intermediate_results(tl, ta, vl, va):
        print('\t\tTraining loss:', tl)
        print('\t\tTraining accuracy:', ta)
        print('\t\tValidation loss:', vl)
        print('\t\tValidation accuracy:', va)
    
    # Initial results
    print('Calculating initial results.')
    results = Results()
    tl, ta = eval_discriminator_epoch(training_dataloader, generator, discriminator, discriminator_loss, device)
    vl, va = eval_discriminator_epoch(validation_dataloader, generator, discriminator, discriminator_loss, device)
    results.update({'discriminator': {'train_loss': tl, 'train_acc': ta, 'val_loss': vl, 'val_acc': va}})
    tl, ta = eval_generator_epoch(training_dataloader, generator, discriminator, generator_loss, device)
    vl, va = eval_generator_epoch(validation_dataloader, generator, discriminator, generator_loss, device)
    results.update({'generator': {'train_loss': tl, 'train_acc': ta, 'val_loss': vl, 'val_acc': va}})
    print('\tDone.')
    print('\tDiscriminator:')
    print_intermediate_results(tl, ta, vl, va)
    print('\tGenerator:')
    print_intermediate_results(tl, ta, vl, va)
    print()
    
    # Discriminator pretraining
    print('Pretraining discriminator.')
    print('\tInitial performamce')
    tl, ta = discriminator_pretrain_eval_epoch(training_dataloader, discriminator, discriminator_loss, device)
    vl, va = discriminator_pretrain_eval_epoch(validation_dataloader, discriminator, discriminator_loss, device)
    results.update({'discriminator': {'pretrain_train_loss': tl, 'pretrain_train_acc': ta, 'pretrain_val_loss': vl, 'pretrain_val_acc': va}})
    print_intermediate_results(tl, ta, vl, va)
    print()
    
    for epoch in range(discriminator_pretraining_epochs):
        print('\tEpoch', epoch+1)
        tl, ta = discriminator_pretrain_epoch(training_dataloader, discriminator, discriminator_loss, discriminator_optimizer, device)
        vl, va = discriminator_pretrain_eval_epoch(validation_dataloader, discriminator, discriminator_loss, device)
        results.update({'discriminator': {'pretrain_train_loss': tl, 'pretrain_train_acc': ta, 'pretrain_val_loss': vl, 'pretrain_val_acc': va}})
        print_intermediate_results(tl, ta, vl, va)
        print()
    print()
    
    # Generator pretraining
    print('Pretraining generator.')
    print('\tInitial performance')
    tl = generator_eval_autoencoder_epoch(training_dataloader, generator, device)
    vl = generator_eval_autoencoder_epoch(validation_dataloader, generator, device)
    results.update({'generator': {'pretrain_train_loss': tl, 'pretrain_val_loss': vl}})
    print_intermediate_results(tl, np.nan, vl, np.nan)
    print()
    
    for epoch in range(generator_pretraining_epochs):
        print('\tEpoch', epoch+1)
        tl = generator_pretrain_autoencoder_epoch(training_dataloader, generator, generator_optimizer, device)
        vl = generator_eval_autoencoder_epoch(validation_dataloader, generator, device)
        results.update({'generator': {'pretrain_train_loss': tl, 'pretrain_val_loss': vl}})
        print_intermediate_results(tl, np.nan, vl, np.nan)
        print()
    print()
    
    # GAN training
    print('Training discriminator and generator simultaneously.')
    for epoch in range(gan_training_epochs):
        print('\tEpoch', epoch+1)
        tgl, tga, tdl, tda = train_gan_epoch(training_dataloader, generator, discriminator, generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer, device)
        vgl, vga = eval_generator_epoch(validation_dataloader, generator, discriminator, generator_loss, device)
        vdl, vda = eval_discriminator_epoch(validation_dataloader, generator, discriminator, discriminator_loss, device)
        results.update({'generator': {'train_loss': tgl, 'train_acc': tga, 'val_loss': vgl, 'val_acc': vga},
                        'discriminator': {'train_loss': tdl, 'train_acc': tda, 'val_loss': vdl, 'val_acc': vda}})
        print('\tDiscriminator results:')
        print_intermediate_results(tdl, tda, vdl, vda)
        print('\tGenerator results:')
        print_intermediate_results(tgl, tga, vgl, vga)
        print()
    print()
    
    # Retrain discriminator from scratch to evaluate GAN performance
    print('Training new discriminator on static trained discriminator.')
    discriminator = _construct_discriminator(dataset, discriminator_constructor, discriminator_kwargs)
    discriminator = discriminator.to(device)
    discriminator_optimizer = discriminator_optimizer_constructor(discriminator.parameters(), **discriminator_optimizer_kwargs)
    
    print('\tInitial performance')
    tl, ta = eval_discriminator_epoch(training_dataloader, generator, discriminator, discriminator_loss, device)
    vl, va = eval_discriminator_epoch(validation_dataloader, generator, discriminator, discriminator_loss, device)
    results.update({'discriminator': {'posttrain_train_loss': tl, 'posttrain_train_acc': ta, 'posttrain_val_loss': vl, 'posttrain_val_acc': va}})
    print_intermediate_results(tl, ta, vl, va)
    print()
    
    for epoch in range(discriminator_posttraining_epochs):
        print('\tEpoch', epoch+1)
        tl, ta = train_discriminator_epoch(training_dataloader, generator, discriminator, discriminator_loss, discriminator_optimizer, device)
        vl, va = eval_discriminator_epoch(validation_dataloader, generator, discriminator, discriminator_loss, device)
        results.update({'discriminator': {'posttrain_train_loss': tl, 'posttrain_train_acc': ta, 'posttrain_val_loss': vl, 'posttrain_val_acc': va}})
        print_intermediate_results(tl, ta, vl, va)
        print()
    print()
    print(results)
    
    return results.data
    