import time
import random
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch import multiprocessing as mp
from training.custom_loss_functions import BasicWrapper

def construct(constructor, *args, **kwargs):
    if constructor is None:
        return None
    else:
        return constructor(*args, **kwargs)

def train_none(disc, disc_loss_fn, disc_opt, train_dataloader, test_dataloader, device, n_epochs, suppress_output, save_dir, metric_fns={}, save_model_period=None):
    from training.train_single_model import train_epoch, eval_epoch
    current_epoch = 0
    def run_epoch():
        nonlocal current_epoch
        t0 = time.time()
        if current_epoch == 0:
            train_results = eval_epoch(train_dataloader, disc, disc_loss_fn, device, metric_fns=metric_fns)
        else:
            train_results = train_epoch(train_dataloader, disc, disc_loss_fn, disc_opt, device, metric_fns=metric_fns)
        test_results = eval_epoch(test_dataloader, disc, disc_loss_fn, device, metric_fns=metric_fns)
        if not suppress_output:
            print('Finished epoch {} in {} seconds.'.format(current_epoch, time.time()-t0))
            #print('Train results: {}'.format(train_results))
            #print('Test results: {}'.format(test_results))
            #print('\n')
        for key, item in train_results.items():
            train_results[key] = np.mean(item, axis=0)
        for key, item in test_results.items():
            test_results[key] = np.mean(item, axis=0)
        if save_dir is not None:
            with open(os.path.join('.', 'results', save_dir, 'train_res_{}.pickle'.format(current_epoch)), 'wb') as F:
                pickle.dump(train_results, F)
            with open(os.path.join('.', 'results', save_dir, 'test_res_{}.pickle'.format(current_epoch)), 'wb') as F:
                pickle.dump(test_results, F)
            if save_model_period is not None and (current_epoch%save_model_period==0 or current_epoch == n_epochs):
                torch.save(disc.state_dict(),
                           os.path.join('.', 'results', save_dir, 'disc_params_{}.pth'.format(current_epoch)))
                if hasattr(disc_opt, 'consolidate_state_dict'):
                    disc_opt.consolidate_state_dict()
                torch.save(disc_opt.state_dict(),
                           os.path.join('.', 'results', save_dir, 'disc_opt_params_{}.pth'.format(current_epoch)))
        current_epoch += 1
    while current_epoch <= n_epochs:
        run_epoch()

def train_randn():
    pass

def train_autoencoder():
    pass

def train_gan():
    pass

def run_trial(
    protection_method,
    disc_constructor=None, disc_kwargs={}, # discriminator model
    disc_loss_constructor=None, disc_loss_kwargs={}, # discriminator loss function
    disc_opt_constructor=None, disc_opt_kwargs={}, # discriminator optimizer
    disc_step_kwargs={}, # kwargs to be passed to the discriminator train step function
    gen_constructor=None, gen_kwargs={}, # generator model
    gen_loss_constructor=None, gen_loss_kwargs={}, # generator loss function
    gen_opt_constructor=None, gen_opt_kwargs={}, # generator optimizer
    gen_step_kwargs={}, # kwargs to be passed to the generator train step function
    aux_disc_constructor=None, aux_disc_kwargs={}, # independent discriminator which will be trained periodically against fixed defense
    aux_disc_loss_constructor=None, aux_disc_loss_kwargs={}, # loss function for above
    aux_disc_opt_constructor=None, aux_disc_opt_kwargs={}, # optimizer for above
    dataset_constructor=None, train_dataset_kwargs={}, test_dataset_kwargs={}, dataloader_kwargs={}, # dataset and its arguments
    trial_kwargs={}, # Miscellaneous trial arguments -- number of epochs, rate of independent disc training, etc.
    device=None,
    suppress_output=False,
    save_dir=None
    ):
    # Generate the seed which will be used for all processes
    print('Constructing trial processes.')
    
    if 'seed' in trial_kwargs.keys():
        seed = trial_kwargs['seed']
    else:
        seed = time.time_ns() & 0xFFFFFFFF
    trial_kwargs['seed'] = seed
    print('Seed: {}'.format(seed))
    
    # Determine number of devices to use
    if device is None:
        if cuda.is_available():
            device_count = torch.cuda.device_count()
        else:
            device_count = 0
    else:
        if device == 'cpu':
            device_count = 0
        elif device == 'cuda':
            device_count = torch.cuda.device_count()
        else:
            assert False
    print('Device count: {}'.format(device_count))
          
    trial_args = [protection_method, disc_constructor, disc_kwargs, disc_loss_constructor, disc_loss_kwargs, disc_opt_constructor, disc_opt_kwargs, disc_step_kwargs, gen_constructor, gen_kwargs, gen_loss_constructor, gen_loss_kwargs, gen_opt_constructor, gen_opt_kwargs, gen_step_kwargs, aux_disc_constructor, aux_disc_kwargs, aux_disc_loss_constructor, aux_disc_loss_kwargs, aux_disc_opt_constructor, aux_disc_opt_kwargs, dataset_constructor, train_dataset_kwargs, test_dataset_kwargs, dataloader_kwargs, trial_kwargs, device, suppress_output, save_dir]
    
    if device_count >= 1:
        world_size = device_count
        print('Creating {} processes.'.format(world_size))
        mp.spawn(run_trial_process,
                 [world_size, *trial_args],
                 nprocs=world_size,
                 join=True)
    else:
        print('Not using multiprocessing.')
        run_trial_process(0, 0, *trial_args)
        
def run_trial_process(
    rank, world_size,
    protection_method,
    disc_constructor=None, disc_kwargs={}, # discriminator model
    disc_loss_constructor=None, disc_loss_kwargs={}, # discriminator loss function
    disc_opt_constructor=None, disc_opt_kwargs={}, # discriminator optimizer
    disc_step_kwargs={}, # kwargs to be passed to the discriminator train step function
    gen_constructor=None, gen_kwargs={}, # generator model
    gen_loss_constructor=None, gen_loss_kwargs={}, # generator loss function
    gen_opt_constructor=None, gen_opt_kwargs={}, # generator optimizer
    gen_step_kwargs={}, # kwargs to be passed to the generator train step function
    aux_disc_constructor=None, aux_disc_kwargs={}, # independent discriminator which will be trained periodically against fixed defense
    aux_disc_loss_constructor=None, aux_disc_loss_kwargs={}, # loss function for above
    aux_disc_opt_constructor=None, aux_disc_opt_kwargs={}, # optimizer for above
    dataset_constructor=None, train_dataset_kwargs={}, test_dataset_kwargs={}, dataloader_kwargs={}, # dataset and its arguments
    trial_kwargs={}, # Miscellaneous trial arguments -- number of epochs, rate of independent disc training, etc.
    device=None,
    suppress_output=False,
    save_dir=None
    ): # Device on which to place the models and samples
    
    if not suppress_output:
        print('Process of rank {} created.'.format(rank))
    suppress_output = suppress_output or rank!=0
    
    if world_size != 0:
        dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=world_size)
        device = rank
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    
    assert protection_method in ['none', 'randnoise', 'autoencoder', 'gan']
    if save_dir is not None:
        os.makedirs(os.path.join('.', 'results', save_dir), exist_ok=True)
    
    random.seed(trial_kwargs['seed'])
    np.random.seed(trial_kwargs['seed'])
    torch.manual_seed(trial_kwargs['seed'])
    
    if 'n_epochs' in trial_kwargs.keys():
        n_epochs = trial_kwargs['n_epochs']
    else:
        n_epochs = 1
    
    train_dataset = construct(dataset_constructor, train=True, **train_dataset_kwargs)
    test_dataset  = construct(dataset_constructor, train=False, **test_dataset_kwargs)
    if world_size != 0:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, num_replicas=world_size, rank=rank)
        test_sampler  = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = construct(DataLoader, train_dataset,
                                 sampler=None if world_size==0 else train_sampler,
                                 shuffle=True if world_size==0 else False, **dataloader_kwargs)
    test_dataloader  = construct(DataLoader, test_dataset,
                                 sampler=None if world_size==0 else test_sampler,
                                 shuffle=False, **dataloader_kwargs)
    
    disc         = construct(disc_constructor, train_dataset.trace_shape, **disc_kwargs)
    if disc is not None:
        if world_size != 0:
            disc.cuda(device)
            disc = nn.parallel.DistributedDataParallel(disc, device_ids=[device])
    disc_loss_fn = construct(disc_loss_constructor, **disc_loss_kwargs)
    if disc is not None and trial_kwargs['use_zero_redundancy_optimizer']:
        disc_opt = ZeroRedundancyOptimizer(
            disc.parameters(),
            disc_opt_constructor,
            **disc_opt_kwargs)
    else:
        disc_opt = construct(disc_opt_constructor, disc.parameters() if disc is not None else None, **disc_opt_kwargs)
    gen          = construct(gen_constructor, train_dataset.trace_shape, **gen_kwargs)
    if gen is not None:
        if world_size != 0:
            gen.cuda(device)
            gen = nn.parallel.DistributedDataParallel(gen, device_ids=[device])
    gen_loss_fn  = construct(gen_loss_constructor, **gen_loss_kwargs)
    if gen is not None and trial_kwargs['use_zero_redundancy_optimizer']:
        gen_opt = ZeroRedundancyOptimizer(
            gen.parameters(),
            gen_opt_constructor,
            **gen_opt_kwargs)
    else:
        gen_opt = construct(gen_opt_constructor, gen.parameters() if gen is not None else None, **gen_opt_kwargs)
    eg_trace, eg_label, _ = next(iter(train_dataloader))
    if disc_loss_fn is not None:
        try:
            _ = disc_loss_fn(disc(eg_trace), eg_trace, eg_label)
        except TypeError:
            disc_loss_fn = BasicWrapper(disc_loss_fn)
    if gen_loss_fn is not None:
        try:
            _ = gen_loss_fn(gen(eg_trace), eg_trace, eg_label)
        except TypeError:
            gen_loss_fn = BasicWrapper(gen_loss_fn)
    
    assert train_dataset is not None
    assert test_dataset is not None
    assert train_dataloader is not None
    assert test_dataloader is not None
    if protection_method in ['none', 'randnoise']:
        assert disc is not None
        assert disc_loss_fn is not None
        assert disc_opt is not None
        assert gen is None
        assert gen_loss_fn is None
        assert gen_opt is None
    elif protection_method in ['autoencoder']:
        assert disc is None
        assert disc_loss_fn is None
        assert disc_opt is None
        assert gen is not None
        assert gen_loss_fn is not None
        assert gen_opt is not None
    elif protection_method in ['gan']:
        assert disc is not None
        assert disc_loss_fn is not None
        assert disc_opt is not None
        assert gen is not None
        assert gen_loss_fn is not None
        assert gen_opt is not None
    
    if not suppress_output:
        print('Beginning training.')
        print('Protection method: {}'.format(protection_method))
        print('Random seed: {}'.format(trial_kwargs['seed']))
        print('Number of epochs: {}'.format(n_epochs))
        print('Device: {}'.format('cuda:%d'%(device) if world_size != 0 else device))
        print('Save directory: {}'.format(save_dir))
        print('Disc: {}'.format(disc))
        print('Disc opt: {}'.format(disc_opt))
        print('Disc loss fn: {}'.format(disc_loss_fn))
        print('Gen: {}'.format(gen))
        print('Gen opt: {}'.format(gen_opt))
        print('Gen loss fn: {}'.format(gen_loss_fn))
        print('Train dataset: {}'.format(train_dataset))
        print('Train dataloader: {}'.format(train_dataloader))
        print('Test dataset: {}'.format(test_dataset))
        print('Test dataloader: {}'.format(test_dataloader))
        print('\n')
    
    if protection_method == 'none':
        train_none(disc, disc_loss_fn, disc_opt, train_dataloader, test_dataloader, device, n_epochs,
                   suppress_output, save_dir if rank==0 else None, metric_fns=trial_kwargs['metric_fns'],
                   save_model_period=trial_kwargs['save_model_period'])