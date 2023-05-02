import time
import argparse
import random
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
from training.common import *
from training.multitask_sca import *
from models.multitask_resnet1d import Classifier
from models.stargan2_architecture import Generator, Discriminator as LeakageDiscriminator
from models.averaged_model import get_averaged_model
from datasets.google_scaaml import GoogleScaamlDataset

VALID_TARGET_REPR = [
    'bits',
    'bytes',
    'hamming_weight'
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

@torch.no_grad()
def int_to_bytes(x):
    assert x.dtype == torch.long
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x
    
@torch.no_grad()
def int_to_binary(x):
    assert x.dtype == torch.long
    x = x.clone()
    out = torch.zeros(x.size(0), 8, dtype=torch.float, device=x.device)
    for n in range(7, -1, -1):
        high = torch.ge(x, 2**n)
        out[:, n] = high
        x -= high*2**n
    return out

@torch.no_grad()
def int_to_hamming_weight(x):
    assert x.dtype == torch.long
    x_bin = int_to_binary(x)
    x_hw = torch.sum(x_bin, dim=-1).to(torch.long)
    return x_hw

@torch.no_grad()
def acc_bytes(x, y):
    x, y = val(x), val(y)
    return np.mean(np.equal(np.argmax(x, axis=-1), y))

@torch.no_grad()
def acc_bits(x, y):
    x, y = val(x), val(y)
    return np.mean(np.equal(x>0, y))

@torch.no_grad()
def acc_hw(x, y):
    x, y = val(x), val(y)
    return np.mean(np.equal(np.argmax(x, axis=-1), y))

def loss_bytes(x, y):
    return nn.functional.cross_entropy(x, y)

def loss_bits(x, y):
    return nn.functional.binary_cross_entropy(torch.sigmoid(x), y)

def loss_hw(x, y):
    return nn.functional.cross_entropy(x, y)
    
def run_trial(
    dataset=GoogleScaamlDataset,
    dataset_kwargs={},
    gen_constructor=Generator,
    gen_kwargs={},
    eval_classifier_constructor=Classifier,
    gen_opt=optim.Adam,
    gen_opt_kwargs={'lr': 1e-4, 'betas': (0.5, 0.999)},
    disc_constructor=LeakageDiscriminator,
    disc_kwargs={},
    disc_opt=optim.Adam,
    disc_opt_kwargs={'lr': 4e-4, 'betas': (0.5, 0.999)},
    disc_steps_per_gen_step=5.0,
    target_repr='hamming_weight',
    target_bytes='all',
    target_attack_pts='sub_bytes_in',
    signal_length=20000, crop_length=20000, downsample_ratio=4, noise_scale=0.0,
    epochs=100,
    device=None,
    pretrain=False,
    posttrain_epochs=25,
    batch_size=32,
    l1_rec_coefficient=0.0,
    gen_classification_coefficient=1.0,
    average_deviation_penalty=1e-1,
    average_update_coefficient=1e-4,
    calculate_weight_norms=True,
    calculate_grad_norms=True,
    save_dir=None, trial_info=None):
    
    if pretrain:
        gen_opt = optim.Adam
        gen_opt_kwargs = {'lr': 2e-4}
        disc_opt = optim.Adam
        disc_opt_kwargs = {'lr': 2e-4}
        disc_steps_per_gen_step = 1.0
        average_deviation_penalty = 0.0
    
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
    
    if target_bytes == 'all':
        target_bytes = VALID_TARGET_BYTES
    if target_attack_pts == 'all':
        target_attack_pts = VALID_ATTACK_PTS
    if not isinstance(target_bytes, list):
        target_bytes = [target_bytes]
    if not isinstance(target_attack_pts, list):
        target_attack_pts = [target_attack_pts]
    assert all(x in VALID_TARGET_BYTES for x in target_bytes)
    assert all(x in VALID_ATTACK_PTS for x in target_attack_pts)
    
    train_transform = SignalTransform(signal_length//downsample_ratio, crop_length//downsample_ratio, noise_scale)
    train_dataset = dataset(transform=train_transform, train=True, download=True, whiten_traces=True,
                                        interval_to_use=[0, signal_length], downsample_ratio=downsample_ratio,
                                        bytes=target_bytes, store_in_ram=False, attack_points=target_attack_pts)
    test_dataset = dataset(transform=None, train=False, download=True, whiten_traces=True,
                                       interval_to_use=[0, crop_length], downsample_ratio=downsample_ratio,
                                       bytes=target_bytes, store_in_ram=False, attack_points=target_attack_pts)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True)
    print('Train dataset:')
    print(train_dataset)
    print('\n\n')
    print('Test dataset:')
    print(test_dataset)
    print('\n\n')
    
    head_sizes = OrderedDict({})
    for tap in target_attack_pts:
        for tb in target_bytes:
            head_name = '{}__{}__{}'.format(target_repr, tap, tb)
            head_sizes[head_name] = {
                'bits': 8,
                'hamming_weight': 9,
                'bytes': 256
            }[target_repr]
                
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
    
    print(gen)
    print(disc)
    print()
    print('Generator parameters:', sum(p.numel() for p in gen.parameters() if p.requires_grad))
    print('Discriminator parameters:', sum(p.numel() for p in disc.parameters() if p.requires_grad))
    
    eval_head_sizes = OrderedDict({})
    for tap in target_attack_pts:
        for tb in target_bytes:
            head_name = '{}__{}__{}'.format('bytes', tap, tb)
            eval_head_sizes[head_name] = 256
    
    eval_classifier = eval_classifier_constructor((1, crop_length//downsample_ratio), eval_head_sizes)
    eval_classifier_state = torch.load(os.path.join('.', 'trained_models', 'google_scaaml_classifier.pth'))
    eval_classifier.load_state_dict(eval_classifier_state)
    eval_classifier = eval_classifier.to(device)
    
    to_repr_fn = {
        'bytes': (int_to_bytes, acc_bytes, loss_bytes),
        'bits': (int_to_binary, acc_bits, loss_bits),
        'hamming_weight': (int_to_hamming_weight, acc_hw, loss_hw)
    }[target_repr]
    
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
            'pretrain': pretrain,
            'gen_opt_scheduler': gen_opt_scheduler,
            'disc_opt_scheduler': disc_opt_scheduler,
            'l1_rec_coefficient': l1_rec_coefficient,
            'gen_classification_coefficient': gen_classification_coefficient,
            'return_weight_norms': calculate_weight_norms,
            'return_grad_norms': calculate_grad_norms,
            'average_deviation_penalty': average_deviation_penalty,
            'disc_steps_per_gen_step': disc_steps_per_gen_step,
            'leakage_eval_disc': eval_classifier,
            'to_repr_fn': to_repr_fn
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
            images = [test_rv['orig_example'][0], test_rv['rec_example'][0]]
            if not pretrain:
                images.append(test_rv['crec_example'][0])
            for *egs, ax in zip(*images, axes.flatten()):
                orig_eg, rec_eg = egs[0], egs[1]
                rec_diff = rec_eg - orig_eg
                ax.plot(rec_diff.flatten(), linestyle='none', marker='.', color='blue')
                if not pretrain:
                    crec_eg = egs[2]
                    crec_diff = crec_eg - orig_eg
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
    
    for epoch_idx in range(1, epochs+1):
        update_results(epoch_idx)
    
    if pretrain:
        model_save_dir = os.path.join('.', 'trained_models')
        os.makedirs(model_save_dir, exist_ok=True)
        gen_state_dict = {k: v.cpu() for k, v in gen.state_dict().items()}
        torch.save(gen_state_dict, os.path.join(model_save_dir, 'pretrained_generator.pth'))
        disc_state_dict = {k: v.cpu() for k, v in disc.state_dict().items()}
        torch.save(disc_state_dict, os.path.join(model_save_dir, 'pretrained_discriminator.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int, help='Random seed to use for this trial.')
    parser.add_argument('--device', default=None, type=str, help='Device to use for this trial.')
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    run_trial(pretrain=True, device=args.device)