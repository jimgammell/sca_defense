import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from training.custom_loss_functions import BasicWrapper

def construct(constructor, *args, **kwargs):
    if constructor is None:
        return None
    else:
        return constructor(*args, **kwargs)

def train_none(disc, disc_loss_fn, disc_opt, train_dataloader, test_dataloader, device, n_epochs, suppress_output, save_dir):
    from training.train_single_model import train_epoch, eval_epoch
    current_epoch = 0
    def run_epoch():
        nonlocal current_epoch
        if current_epoch == 0:
            train_results = eval_epoch(train_dataloader, disc, disc_loss_fn, device)
        else:
            train_results = train_epoch(train_dataloader, disc, disc_loss_fn, disc_opt, device)
        test_results = eval_epoch(test_dataloader, disc, disc_loss_fn, device)
        if not suppress_output:
            print('Finished epoch {}'.format(current_epoch))
            print('Train results: {}'.format(train_results))
            print('Test results: {}'.format(test_results))
            print('\n')
        if save_dir is not None:
            with open(os.path.join(save_dir, 'train_res_{}.pickle'.format(current_epoch)), 'wb') as F:
                pickle.dump(train_results, F)
            with open(os.path.join(save_dir, 'test_res_{}.pickle'.format(current_epoch)), 'wb') as F:
                pickle.dump(test_results, F)
            torch.save(disc.state_dict(), os.path.join(save_dir, 'disc_params_{}.pth'.format(current_epoch)))
            torch.save(disc_opt.state_dict(), os.path.join(save_dir, 'disc_opt_params_{}.pth'.format(current_epoch)))
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
    ): # Device on which to place the models and samples
    
    assert protection_method in ['none', 'randnoise', 'autoencoder', 'gan']
    
    if 'seed' in trial_kwargs.keys():
        seed = trial_kwargs['seed']
    else:
        seed = time.time_ns() & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if 'n_epochs' in trial_kwargs.keys():
        n_epochs = trial_kwargs['n_epochs']
    else:
        n_epochs = 1
    
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    train_dataset    = construct(dataset_constructor, train=True, **train_dataset_kwargs)
    test_dataset     = construct(dataset_constructor, train=False, **test_dataset_kwargs)
    train_dataloader = construct(DataLoader, train_dataset, shuffle=True, **dataloader_kwargs)
    test_dataloader  = construct(DataLoader, test_dataset, shuffle=False, **dataloader_kwargs)
    
    disc         = construct(disc_constructor, train_dataset.trace_shape, **disc_kwargs)
    disc_loss_fn = construct(disc_loss_constructor, **disc_loss_kwargs)
    disc_opt     = construct(disc_opt_constructor, disc.parameters() if disc is not None else None, **disc_opt_kwargs)
    gen          = construct(gen_constructor, train_dataset.trace_shape, **gen_kwargs)
    gen_loss_fn  = construct(gen_loss_constructor, **gen_loss_kwargs)
    gen_opt      = construct(gen_opt_constructor, gen.parameters() if gen is not None else None, **gen_opt_kwargs)
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
        print('Random seed: {}'.format(seed))
        print('Number of epochs: {}'.format(n_epochs))
        print('Device: {}'.format(device))
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
        train_none(disc, disc_loss_fn, disc_opt, train_dataloader, test_dataloader, device, n_epochs, suppress_output, save_dir)