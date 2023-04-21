import numpy as np
import time
from copy import deepcopy
import torch
from torch import nn, optim
from datasets.google_scaaml import GoogleScaamlDataset
from models.multitask_resnet1d import Classifier
from training.multitask_sca import train_epoch, eval_epoch

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

def main(
    target_repr='bytes', #'bits',
    target_bytes='all',
    target_attack_pts=['sub_bytes_in', 'sub_bytes_out'],
    signal_length=20000, crop_length=20000, downsample_ratio=4, noise_scale=0.00,
    #signal_length=25000, crop_length=20000, noise_scale=0.01,
    num_epochs=100, weight_decay=0.0, max_lr=1e-2, pct_start=0.3, dropout=0.1,
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
    train_dataset = GoogleScaamlDataset(transform=train_transform, train=True, download=True, whiten_traces=True,
                                        interval_to_use=[0, signal_length], downsample_ratio=downsample_ratio, bytes=target_bytes,
                                        store_in_ram=False, attack_points=target_attack_pts)
    test_dataset = GoogleScaamlDataset(transform=None, train=False, download=True, whiten_traces=True,
                                       interval_to_use=[0, crop_length], downsample_ratio=downsample_ratio, bytes=target_bytes,
                                       store_in_ram=False, attack_points=target_attack_pts)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=32, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=32, num_workers=8)
    print('Train dataset:')
    print(train_dataset)
    print('\n\n')
    print('Test dataset:')
    print(test_dataset)
    print('\n\n')
    
    head_sizes = {}
    for tr in target_repr:
        for tap in target_attack_pts:
            for tb in target_bytes:
                head_name = '{}__{}__{}'.format(tr, tap, tb)
                head_sizes[head_name] = 8 if tr == 'bits' else 256
    classifier = Classifier((1, crop_length), head_sizes, dense_dropout=dropout).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=max_lr, weight_decay=weight_decay)
    #lr_scheduler = optim.lr_scheduler.OneCycleLR(
    #    optimizer, max_lr, epochs=num_epochs, steps_per_epoch=len(train_dataloader), pct_start=pct_start
    #)
    lr_scheduler = None
    print('Model:')
    print(classifier)
    print('\n\n')
    print('Optimizer:')
    print(optimizer)
    print('\n\n')
    print('Learning rate scheduler:')
    print(lr_scheduler)
    print('\n\n')
    
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
    return results, best_state_dict
    
if __name__ == '__main__':
    main()