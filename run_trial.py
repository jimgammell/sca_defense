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
from ray.air.checkpoint import Checkpoint
from ray.air import session

def construct(constructor, *args, **kwargs):
    if constructor is None:
        return None
    else:
        return constructor(*args, **kwargs)
    
def train_single_model(model, loss_fn, optimizer, train_dataloader, test_dataloader, device, n_epochs, suppress_output, save_dir, metric_fns={}, save_model_period=None, using_raytune=False, keys_to_report=None, step_kwargs={}, model_name=None):
    from training.train_single_model import train_epoch, eval_epoch
    if model_name is None:
        model_name = 'model'
    current_epoch = 0
    def run_epoch():
        nonlocal current_epoch, keys_to_report
        t0 = time.time()
        if current_epoch == 0 and not using_raytune:
            train_results = eval_epoch(train_dataloader, model, loss_fn, device, metric_fns=metric_fns)
        else:
            train_results = train_epoch(train_dataloader, model, loss_fn, optimizer, device, metric_fns=metric_fns, **step_kwargs)
        test_results = eval_epoch(test_dataloader, model, loss_fn, device, metric_fns=metric_fns)
        for key, item in train_results.items():
            train_results[key] = np.mean(item, axis=0)
        for key, item in test_results.items():
            test_results[key] = np.mean(item, axis=0)
        if not suppress_output:
            print('Finished epoch {} in {} seconds.'.format(current_epoch, time.time()-t0))
            print('Train results:', train_results)
            print('Test results:', test_results)
        checkpoint = None
        if save_dir is not None:
            with open(os.path.join('.', 'results', save_dir, 'train_res_{}.pickle'.format(current_epoch)), 'wb') as F:
                pickle.dump(train_results, F)
            with open(os.path.join('.', 'results', save_dir, 'test_res_{}.pickle'.format(current_epoch)), 'wb') as F:
                pickle.dump(test_results, F)
            if save_model_period is not None and (current_epoch%save_model_period==0 or current_epoch==n_epochs):
                current_save_dir = os.path.join('.', 'results', save_dir, 'checkpoint_{}'.format(current_epoch))
                os.makedirs(current_save_dir, exist_ok=True)
                torch.save(model.state_dict(),
                           os.path.join(current_save_dir, model_name+'_state.pth'))
                if hasattr(optimizer, 'consolidate_state_dict'):
                    optimizer.consolidate_state_dict()
                torch.save(optimizer.state_dict(),
                           os.path.join(current_save_dir, model_name+'_opt_state.pth'))
                if using_raytune:
                    checkpoint = Checkpoint.from_directory(current_save_dir)
        if using_raytune:
            if keys_to_report is None:
                keys_to_report = list(train_results.keys())
                assert all(k in test_results.keys() for k in keys_to_report)
                assert all(k in keys_to_report for k in test_results.keys())
            results = dict(**{'train_'+key: item for key, item in train_results.items() if key in keys_to_report},
                           **{'test_'+key: item for key, item in test_results.items() if key in keys_to_report})
            session.report(results, checkpoint=checkpoint if checkpoint is not None else None)
        current_epoch += 1
    while using_raytune or current_epoch <= n_epochs:
        run_epoch()
    
def train_none(disc, disc_loss_fn, disc_opt, train_dataloader, test_dataloader, device, n_epochs, suppress_output, save_dir, metric_fns={}, save_model_period=None, using_raytune=False, keys_to_report=None, disc_step_kwargs={}):
    train_single_model(disc, disc_loss_fn, disc_opt, train_dataloader, test_dataloader, device, n_epochs, suppress_output, save_dir, metric_fns, save_model_period, using_raytune, keys_to_report, disc_step_kwargs, model_name='disc')
def train_autoencoder(gen, gen_loss_fn, gen_opt, train_dataloader, test_dataloader, device, n_epochs, suppress_output, save_dir, metric_fns={}, save_model_period=None, using_raytune=False, keys_to_report=None, gen_step_kwargs={}):
    train_single_model(gen, gen_loss_fn, gen_opt, train_dataloader, test_dataloader, device, n_epochs, suppress_output, save_dir, metric_fns, save_model_period, using_raytune, keys_to_report, gen_step_kwargs, model_name='gen')

def train_randn():
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
    save_dir=None,
    using_raytune=False
    ):
    # Generate the seed which will be used for all processes
    
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
          
    trial_args = [protection_method, disc_constructor, disc_kwargs, disc_loss_constructor, disc_loss_kwargs, disc_opt_constructor, disc_opt_kwargs, disc_step_kwargs, gen_constructor, gen_kwargs, gen_loss_constructor, gen_loss_kwargs, gen_opt_constructor, gen_opt_kwargs, gen_step_kwargs, aux_disc_constructor, aux_disc_kwargs, aux_disc_loss_constructor, aux_disc_loss_kwargs, aux_disc_opt_constructor, aux_disc_opt_kwargs, dataset_constructor, train_dataset_kwargs, test_dataset_kwargs, dataloader_kwargs, trial_kwargs, device, suppress_output, save_dir, using_raytune]
    
    if device_count > 1:
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
    save_dir=None,
    using_raytune=False
    ): # Device on which to place the models and samples
    
    if not suppress_output:
        print('Process of rank {} created.'.format(rank))
        
    suppress_output = suppress_output or rank!=0 or using_raytune

    if world_size != 0:
        dist.init_process_group('nccl', init_method='env://', rank=rank, world_size=world_size)
        device = rank
        torch.cuda.set_device(device)
    else:
        pass#device = 'cpu'
    
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
    
    disc         = construct(disc_constructor, train_dataset.x_shape, **disc_kwargs)
    if disc is not None:
        if world_size != 0:
            disc.cuda(device)
            disc = nn.parallel.DistributedDataParallel(disc, device_ids=[device])
        else:
            disc = disc.to(device)
    disc_loss_fn = construct(disc_loss_constructor, **disc_loss_kwargs)
    if disc is not None and trial_kwargs['use_zero_redundancy_optimizer']:
        disc_opt = ZeroRedundancyOptimizer(
            disc.parameters(),
            disc_opt_constructor,
            **disc_opt_kwargs)
    else:
        disc_opt = construct(disc_opt_constructor, disc.parameters() if disc is not None else None, **disc_opt_kwargs)
        
    gen          = construct(gen_constructor, train_dataset.x_shape, **gen_kwargs)
    if gen is not None:
        if world_size != 0:
            gen.cuda(device)
            gen = nn.parallel.DistributedDataParallel(gen, device_ids=[device])
        else:
            gen = gen.to(device)
    gen_loss_fn  = construct(gen_loss_constructor, **gen_loss_kwargs)
    if gen is not None and trial_kwargs['use_zero_redundancy_optimizer']:
        gen_opt = ZeroRedundancyOptimizer(
            gen.parameters(),
            gen_opt_constructor,
            **gen_opt_kwargs)
    else:
        gen_opt = construct(gen_opt_constructor, gen.parameters() if gen is not None else None, **gen_opt_kwargs)
    
    eg_trace, eg_label, _ = next(iter(train_dataloader))
    eg_trace = eg_trace.to(device)
    eg_label = eg_label.to(device)
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
    
    if using_raytune:
        loaded_checkpoint = session.get_checkpoint()
        if loaded_checkpoint:
            with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
                if os.path.exists(os.path.join(loaded_checkpoint_dir, 'disc_state.pth')):
                    disc_state = torch.load(os.path.join(loaded_checkpoint_dir, 'disc_state.pth'))
                    disc.load_state_dict(disc_state)
                if os.path.exists(os.path.join(loaded_checkpoint_dir, 'disc_opt_state.pth')):
                    disc_opt_state = torch.load(os.path.join(loaded_checkpoint_dir, 'disc_opt_state.pth'))
                    disc_opt.load(disc_opt_state)
                if os.path.exists(os.path.join(loaded_checkpoint_dir, 'gen_state.pth')):
                    gen_state = torch.load(os.path.join(loaded_checkpoint_dir, 'gen_state.pth'))
                    gen.load(gen_state)
                if os.path.exists(os.path.join(loaded_checkpoint_dir, 'gen_opt_state.pth')):
                    gen_opt_state = torch.load(os.path.join(loaded_checkpoint_dir, 'gen_opt_state.pth'))
                    gen_opt.load(gen_opt_state)
    
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
                   suppress_output, save_dir if rank==0 else None, 
                   metric_fns=None if 'metric_fns' not in trial_kwargs.keys() else trial_kwargs['metric_fns'],
                   save_model_period=trial_kwargs['save_model_period'], using_raytune=using_raytune, disc_step_kwargs=disc_step_kwargs)
    elif protection_method == 'autoencoder':
        train_autoencoder(gen, gen_loss_fn, gen_opt, train_dataloader, test_dataloader, device, n_epochs,
                          suppress_output, save_dir if rank==0 else None,
                          metric_fns=None if 'metric_fns' not in trial_kwargs.keys() else trial_kwargs['metric_fns'],
                          save_model_period=trial_kwargs['save_model_period'], using_raytune=using_raytune, gen_step_kwargs=gen_step_kwargs)
    else:
        assert False
