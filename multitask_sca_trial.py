import time
import os
from collections import OrderedDict
import sys
from copy import deepcopy
import pickle
import numpy as np
from matplotlib import pyplot as plt
import imageio
import torch
from torch import nn, optim
from training.multitask_sca import *
from models.multitask_resnet1d import *
from models.averaged_model import get_averaged_model
from datasets.google_scaaml import GoogleScaamlDataset

VALID_TARGET_REPR = [
    'bits',
    'bytes'
]
VALID_TARGET_BYTES = [
    *list(range(16))
]
VALID_ATTACK_PTS = [
    'sub_bytes_in',
    'sub_bytes_out',
    'key'
]

class SignalTransform(nn.Module):
    def __init__(self, input_length, cropped_length, noise_scale):
        super().__init__()
        
        self.input_length = input_length
        self.cropped_length = cropped_length
        self.noise_scale = noise_scale
        
    @torch.no_grad()
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if self.input_length > self.cropped_length:
            crop_start = torch.randint(self.input_length-self.cropped_length, size=(1,))
            x = x[:, crop_start:crop_start+self.cropped_length]
        noise = torch.randn_like(x)*self.noise_scale
        x = x + noise
        return x

def run_trial(
    dataset=GoogleScaamlDataset,
    dataset_kwargs={},
    gen_constructor=Generator,
    gen_kwargs={},
    gen_opt=optim.Adam,
    gen_opt_kwargs={'lr': 1e-4, 'betas': (0.5, 0.999)},
    disc_constructor=LeakageDiscriminator,
    disc_kwargs={},
    disc_opt=optim.Adam,
    disc_opt_kwargs={'lr': 4e-4, 'betas': (0.5, 0.999)},
    disc_steps_per_gen_step=3.0,
    target_repr='bits',
    target_bytes='all',
    target_attack_pts='sub_bytes_in',
    signal_length=20000, crop_length=20000, downsample_ratio=4, noise_scale=0.0,
    epochs=100,
    device=None,
    posttrain_epochs=25,
    batch_size=32,
    l1_rec_coefficient=0.0,
    gen_classification_coefficient=1.0,
    average_deviation_penalty=1e-1,
    average_update_coefficient=1e-4,
    calculate_weight_norms=True,
    calculate_grad_norms=True,
    save_dir=None, trial_info=None):
    
    if save_dir is None:
        save_dir = os.path.join('.', 'results', 'multitask_sca_trial')
    eg_frames_dir = os.path.join(save_dir, 'eg_frames')
    results_dir = os.path.join(save_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eg_frames_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    for s in ['train', 'test']:
        os.makedirs(os.path.join(results_dir, s), exist_ok=True)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if target_repr == 'all':
        target_repr = VALID_TARGET_REPR
    if target_bytes == 'all':
        target_bytes = VALID_TARGET_BYTES
    if target_attack_pts == 'all':
        target_attack_pts = VALID_ATTACK_PTS
    if not isinstance(target_repr, list):
        target_repr = [target_repr]
    if not isinstance(target_bytes, list):
        target_bytes = [target_bytes]
    if not isinstance(target_attack_pts, list):
        target_attack_pts = [target_attack_pts]
    assert all(x in VALID_TARGET_REPR for x in target_repr)
    assert all(x in VALID_TARGET_BYTES for x in target_bytes)
    assert all(x in VALID_ATTACK_PTS for x in target_attack_pts)
    
    train_transform = SignalTransform(signal_length//downsample_ratio, crop_length//downsample_ratio, noise_scale)
    train_dataset = dataset(transform=train_transform, train=True, download=True, whiten_traces=True,
                                        interval_to_use=[0, signal_length], downsample_ratio=downsample_ratio,
                                        bytes=target_bytes, store_in_ram=False, attack_points=target_attack_pts)
    test_dataset = dataset(transform=None, train=False, download=True, whiten_traces=True,
                                       interval_to_use=[0, crop_length], downsample_ratio=downsample_ratio,
                                       bytes=target_bytes, store_in_ram=False, attack_points=target_attack_pts)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=8)
    print('Train dataset:')
    print(train_dataset)
    print('\n\n')
    print('Test dataset:')
    print(test_dataset)
    print('\n\n')
    
    head_sizes = OrderedDict({})
    for tr in target_repr:
        for tap in target_attack_pts:
            for tb in target_bytes:
                head_name = '{}__{}__{}'.format(tr, tap, tb)
                head_sizes[head_name] = 8 if tr == 'bits' else 256
                
    if average_deviation_penalty > 0.0:
        avg_fn = lambda x1, x2: (1-average_update_coefficient)*x1 + average_update_coefficient*x2
        new_gen_constructor = get_averaged_model(gen_constructor, device, avg_fn=avg_fn)
        gen_constructor = new_gen_constructor
    gen = gen_constructor((1, crop_length//downsample_ratio), head_sizes, **gen_kwargs).to(device)
    disc = disc_constructor((1, crop_length//downsample_ratio), head_sizes, **disc_kwargs).to(device)
    gen_opt = gen_opt(gen.parameters(), **gen_opt_kwargs)
    disc_opt = disc_opt(disc.parameters(), **disc_opt_kwargs)
    gen_opt_scheduler = None
    disc_opt_scheduler = None
    
    if trial_info is not None:
        with open(os.path.join(results_dir, 'trial_info.pickle'), 'wb') as F:
            pickle.dump(trial_info, F)
    
    def print_d(d):
        for key, item in d.items():
            if not hasattr(item, '__len__'):
                print('\t{}: {}'.format(key, item))
    
    def update_results(current_epoch):
        print('\n\n')
        print('Starting epoch {}.'.format(current_epoch))
        
        kwargs = {
            'gen_opt_scheduler': gen_opt_scheduler,
            'disc_opt_scheduler': disc_opt_scheduler,
            'l1_rec_coefficient': l1_rec_coefficient,
            'gen_classification_coefficient': gen_classification_coefficient,
            'return_weight_norms': calculate_weight_norms,
            'return_grad_norms': calculate_grad_norms,
            'average_deviation_penalty': average_deviation_penalty,
            'disc_steps_per_gen_step': disc_steps_per_gen_step
        }
        
        t0 = time.time()
        train_rv = train_cyclegan_epoch(train_dataloader, gen, gen_opt, disc, disc_opt, device, **kwargs)
        train_rv['time'] = time.time()-t0
        print('Done with training phase. Results:')
        print_d(train_rv)
        
        t0 = time.time()
        test_rv = eval_cyclegan_epoch(test_dataloader, gen, disc, device, return_example_idx=0, **kwargs)
        test_rv['time'] = time.time()-t0
        print('Done with testing phase. Results:')
        print_d(test_rv)
        
        if any('example' in key for key in test_rv.keys()):
            num_rows = int(np.sqrt(batch_size))
            num_cols = (batch_size//num_rows) + int(batch_size%num_rows!=0)
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
            if len(axes.shape) == 1:
                axes = np.expand_dims(axes, 0)
            for orig_eg, rec_eg, crec_eg, ax in zip(test_rv['orig_example'][0], test_rv['rec_example'][0], test_rv['crec_example'][0], axes.flatten()):
                rec_diff = rec_eg - orig_eg
                crec_diff = crec_eg - orig_eg
                ax.plot(rec_diff.flatten(), linestyle='none', marker='.', color='blue')
                ax.plot(crec_diff.flatten(), linestyle='none', marker='.', color='red')
                ax.set_ylim(-2, 2)
                ax.set_yscale('symlog', linthresh=1e-2)
            fig.suptitle('Epoch {}'.format(current_epoch))
            plt.tight_layout()
            fig.savefig(os.path.join(eg_frames_dir, 'frame_{}.jpg'.format(current_epoch)), dpi=50)
            plt.close()
            for key in deepcopy(list(test_rv.keys())):
                if 'example' in key:
                    del test_rv[key]
        
        with open(os.path.join(results_dir, 'train', 'epoch_{}.pickle'.format(current_epoch)), 'wb') as F:
            pickle.dump(train_rv, F)
        with open(os.path.join(results_dir, 'test', 'epoch_{}.pickle'.format(current_epoch)), 'wb') as F:
            pickle.dump(test_rv, F)
    
    for epoch_idx in range(1, epochs):
        update_results(epoch_idx)
    