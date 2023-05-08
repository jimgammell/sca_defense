import numpy as np
import random
import time
import os
import pickle
import argparse
from copy import deepcopy
import torch
from torch import nn, optim
from datasets.google_scaaml import GoogleScaamlDataset
from models.stargan2_architecture import Classifier
from training.multitask_sca import train_epoch, eval_epoch

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

@torch.no_grad()
def int_to_binary(x):
    assert x.dtype == torch.long
    x = x.clone()
    if x.dim() == 0:
        x = x.view(1)
        non_batched = True
    else:
        non_batched = False
    out = torch.zeros(x.size(0), 8, dtype=torch.float, device=x.device)
    for n in range(7, -1, -1):
        high = torch.ge(x, 2**n)
        out[:, n] = high
        x -= high*2**n
    if non_batched:
        out = out.view(8)
    return out

@torch.no_grad()
def int_to_hamming_weight(x):
    assert x.dtype == torch.long
    x_bin = int_to_binary(x)
    x_hw = torch.sum(x_bin, dim=-1).to(torch.long)
    return x_hw

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

def main(
    target_repr='hamming_weight',
    target_bytes='all',
    classifier_kwargs={},
    target_attack_pts='sub_bytes_in', #['sub_bytes_in', 'sub_bytes_out'],
    signal_length=20000, crop_length=20000, downsample_ratio=4, noise_scale=0.00,
    #signal_length=25000, crop_length=20000, noise_scale=0.01,
    num_epochs=50, weight_decay=1e-4, max_lr=2e-4,
    device=None
):
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
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_transform = SignalTransform(signal_length//downsample_ratio, crop_length//downsample_ratio, noise_scale)
    target_transform = int_to_hamming_weight if 'hamming_weight' in target_repr else None
    train_dataset = GoogleScaamlDataset(transform=train_transform, train=True, download=True, whiten_traces=True,
                                        interval_to_use=[0, signal_length], downsample_ratio=downsample_ratio, bytes=target_bytes,
                                        store_in_ram=False, attack_points=target_attack_pts, target_transform=target_transform)
    test_dataset = GoogleScaamlDataset(transform=None, train=False, download=True, whiten_traces=True,
                                       interval_to_use=[0, crop_length], downsample_ratio=downsample_ratio, bytes=target_bytes,
                                       store_in_ram=False, attack_points=target_attack_pts, target_transform=target_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=8)
    #print('Train dataset:')
    #print(train_dataset)
    #print('\n\n')
    #print('Test dataset:')
    #print(test_dataset)
    #print('\n\n')
    
    head_sizes = {}
    for tr in target_repr:
        for tap in target_attack_pts:
            for tb in target_bytes:
                head_name = '{}__{}__{}'.format(tr, tap, tb)
                head_sizes[head_name] = 8 if tr == 'bits' else 256 if tr == 'bytes' else 9 if tr == 'hamming_weight' else None
    classifier = Classifier((1, crop_length//downsample_ratio), head_sizes, **classifier_kwargs).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=max_lr, weight_decay=weight_decay)
    #lr_scheduler = optim.lr_scheduler.OneCycleLR(
    #    optimizer, max_lr, epochs=num_epochs, steps_per_epoch=len(train_dataloader), pct_start=pct_start
    #)
    lr_scheduler = None
    print('Model with {} parameters:'.format(sum(p.numel() for p in classifier.parameters() if p.requires_grad)))
    print(classifier)
    #print('\n\n')
    #print('Optimizer:')
    #print(optimizer)
    #print('\n\n')
    #print('Learning rate scheduler:')
    #print(lr_scheduler)
    #print('\n\n')
    
    results = {}
    best_state_dict, best_test_acc, epochs_without_improvement = None, -np.inf, 0
    for epoch_idx in range(num_epochs):
        t0 = time.time()
        train_rv = train_epoch(train_dataloader, classifier, optimizer, lr_scheduler, device)
        test_rv = eval_epoch(test_dataloader, classifier, device)
        for key, item in train_rv.items():
            if not 'train_'+key in results.keys():
                results['train_'+key] = []
            results['train_'+key].append(item)
        for key, item in test_rv.items():
            if not 'test_'+key in results.keys():
                results['test_'+key] = []
            results['test_'+key].append(item)
        print('Epoch {} done in {} seconds.'.format(epoch_idx, time.time()-t0))
        def print_stats(phase, metric):
            rv = train_rv if phase == 'train' else test_rv
            print('\t{} {}: {} -- {}'.format(
                phase, metric,
                np.min([item for key, item in rv.items() if metric in key]),
                np.max([item for key, item in rv.items() if metric in key])
            ))
        print_stats('train', 'acc')
        print_stats('train', 'loss')
        print_stats('test', 'acc')
        print_stats('test', 'loss')
        test_acc = np.mean([item for key, item in test_rv.items() if 'acc' in key])
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state_dict = {k: v.cpu() for k, v in deepcopy(classifier).state_dict().items()}
            print('New best model found.')
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print('Epochs without improvement: {}.'.format(epochs_without_improvement))
    model_save_dir = os.path.join('.', 'trained_models', 'google_scaaml_classifier.pth')
    torch.save(best_state_dict, model_save_dir)
    return results, best_state_dict
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int, help='Random seed to use for this trial.')
    parser.add_argument('--device', default=None, type=str, help='Device to use for this trial.')
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    results, best_state_dict = main(device=args.device)
    
    results_dir = os.path.join('.', 'results', 'google_scaaml_classifier')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'results.pickle'), 'wb') as F:
        pickle.dump(results, F)
    model_dir = os.path.join('.', 'trained_models')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(best_state_dict, os.path.join(model_dir, 'google_scaaml_classifier.pth'))