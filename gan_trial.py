import time
import os
import sys
from copy import deepcopy
import pickle
import numpy as np
from matplotlib import pyplot as plt
import imageio
import torch
from torch import nn, optim
from gan_train import *
from models.sagan import Generator, Discriminator
from datasets.classified_mnist import WatermarkedMNIST, ColoredMNIST

def run_trial(
    dataset=ColoredMNIST,
    dataset_kwargs={},
    gen=Generator,
    gen_kwargs={},
    gen_opt=optim.Adam,
    gen_opt_kwargs={'lr': 1e-4, 'betas': (0.0, 0.9)},
    disc=Discriminator,
    disc_kwargs={},
    disc_opt=optim.Adam,
    disc_opt_kwargs={'lr': 4e-4, 'betas': (0.0, 0.9)},
    accelerate_examples=False,
    pretrain_gen_epochs=0,
    epochs=10,
    batch_size=256,
    save_dir=None,
    trial_info=None):
    
    if save_dir is None:
        save_dir = os.path.join('.', 'results', 'gan_trial')
    eg_frames_dir = os.path.join(save_dir, 'eg_frames')
    results_dir = os.path.join(save_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eg_frames_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    for s in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(results_dir, s), exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    mnist_loc = os.path.join('.', 'downloads', 'MNIST')
    train_dataset = dataset(train=True, root=mnist_loc, download=True)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (50000, 10000))
    test_dataset = dataset(train=False, root=mnist_loc, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    gen = gen(dataset.input_shape, **gen_kwargs).to(device)
    gen_opt = gen_opt(gen.parameters(), **gen_opt_kwargs)
    gen_loss_fn = lambda *args: gen_loss(*args, **gen_loss_kwargs)
    disc = disc(dataset.input_shape, **disc_kwargs).to(device)
    disc_opt = disc_opt(disc.parameters(), **disc_opt_kwargs)
    disc_loss_fn = lambda *args: disc_loss(*args, **disc_loss_kwargs)
    train_args = (gen, gen_opt, disc, disc_opt, device)
    eval_args = (gen, disc, device)
    
    results = {}
    def update_results(current_epoch, eval_only=False, autoencoder_gen=False):
        nonlocal results
        def preface_keys(d, preface):
            for key, item in deepcopy(d).items():
                d[preface+'_'+key] = item
                del d[key]
            return d
        def print_d(d):
            for key, item in d.items():
                if not hasattr(item, '__len__'):
                    print('\t{}: {}'.format(key, item))
        print('\n\n')
        print('Starting epoch {}.'.format(current_epoch))
        
        t0 = time.time()
        if not eval_only:
            train_rv = train_epoch(train_dataloader, *train_args,
                                   accelerate_examples=accelerate_examples,
                                   return_weight_norms=True,
                                   return_grad_norms=True,
                                   autoencoder_gen=autoencoder_gen)
        else:
            train_rv = eval_epoch(train_dataloader, *eval_args)
        train_rv['time'] = time.time()-t0
        print('Done with training phase. Results:')
        print_d(train_rv)
        
        t0 = time.time()
        val_rv = eval_epoch(val_dataloader, *eval_args, return_example_idx=0)
        val_rv['time'] = time.time()-t0
        print('Done with validation phase. Results:')
        print_d(val_rv)
        
        t0 = time.time()
        test_rv = eval_epoch(test_dataloader, *eval_args)
        test_rv['time'] = time.time()-t0
        print('Done with test phase. Results:')
        print_d(test_rv)
        
        if 'orig_example' in val_rv.keys() and 'rec_example' in val_rv.keys():
            fig, axes = plt.subplots(4, 10, figsize=(40, 16))
            for eg, ax in zip(val_rv['orig_example'][0].squeeze(), axes[:, :5].flatten()):
                if len(eg.shape) == 3:
                    eg = eg.transpose(1, 2, 0)
                ax.imshow(eg, cmap='binary' if dataset==WatermarkedMNIST else None)
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
            for eg, ax in zip(val_rv['rec_example'][0].squeeze(), axes[:, 5:].flatten()):
                if len(eg.shape) == 3:
                    eg = eg.transpose(1, 2, 0)
                ax.imshow(eg, cmap='binary' if dataset==WatermarkedMNIST else None)
                for spine in ax.spines.values():
                    spine.set_edgecolor('blue')
            for ax in axes.flatten():
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
            fig.suptitle('Epoch: {}'.format(current_epoch))
            plt.tight_layout()
            fig.savefig(os.path.join(eg_frames_dir, 'frame_{}.jpg'.format(current_epoch)))
            plt.close()
            del val_rv['orig_example']
            del val_rv['rec_example']
        
        with open(os.path.join(results_dir, 'train', 'epoch_{}.pickle'.format(current_epoch)), 'wb') as F:
            pickle.dump(train_rv, F)
        with open(os.path.join(results_dir, 'validation', 'epoch_{}.pickle'.format(current_epoch)), 'wb') as F:
            pickle.dump(val_rv, F)
        with open(os.path.join(results_dir, 'test', 'epoch_{}.pickle'.format(current_epoch)), 'wb') as F:
            pickle.dump(test_rv, F)
            
    if trial_info is not None:
        with open(os.path.join(results_dir, 'trial_info.pickle'), 'wb') as F:
            pickle.dump(trial_info, F)
    
    update_results(0, eval_only=True)
    for epoch_idx in range(1, pretrain_gen_epochs+1):
        update_results(epoch_idx, autoencoder_gen=True)
    for epoch_idx in range(pretrain_gen_epochs+1, pretrain_gen_epochs+epochs+1):
        update_results(epoch_idx)
        
def plot_traces(trial_dir):
    figs_dir = os.path.join(trial_dir, 'figures')
    os.makedirs(figs_dir, exist_ok=True)
    
    def get_trace(key, phase):
        epochs, vals = [], []
        for res_path in os.listdir(os.path.join(trial_dir, 'results', phase)):
            with open(os.path.join(trial_dir, 'results', phase, res_path), 'rb') as F:
                results = pickle.load(F)
            if not key in results.keys():
                continue
            val = results[key]
            epoch = int(res_path.split('.')[0].split('_')[-1])
            epochs.append(epoch)
            vals.append(val)
        epochs, vals = np.array(epochs), np.array(vals)
        if len(epochs) == len(vals) == 0:
            return None
        sorted_indices = np.argsort(epochs)
        epochs = epochs[sorted_indices]
        vals = vals[sorted_indices]
        return epochs, vals
    
    ex_acc_tr = get_trace('acceleration_loss', 'train')
    ax_width = 4
    if ex_acc_tr is not None:
        n_axes = 4
    else:
        n_axes = 3
    (fig, axes) = plt.subplots(1, n_axes, figsize=(n_axes*ax_width, ax_width))
    
    axes[0].plot(*get_trace('disc_loss', 'train'), '--', color='red')
    axes[0].plot(*get_trace('disc_loss', 'validation'), '-', color='red', label='Disc loss')
    axes[0].plot(*get_trace('gen_loss', 'train'), '--', color='blue')
    axes[0].plot(*get_trace('gen_loss', 'validation'), '-', color='blue', label='Gen loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('GAN losses over time')
    axes[0].set_yscale('symlog', linthresh=1e-1)
    axes[0].legend()
    
    axes[1].plot(*get_trace('disc_weight_norm', 'train'), '--', color='red', label='Disc (train)')
    axes[1].plot(*get_trace('gen_weight_norm', 'train'), '--', color='blue', label='Gen (train)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Weight norm')
    axes[1].set_title('Weight norms over time')
    axes[1].legend()
    
    axes[2].plot(*get_trace('disc_grad_norm', 'train'), '--', color='red', label='Disc (train)')
    axes[2].plot(*get_trace('gen_grad_norm', 'train'), '--', color='blue', label='Gen (train)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Grad norm')
    axes[2].set_title('Gradient norms over time')
    axes[2].set_yscale('log')
    axes[2].legend()
    
    if ex_acc_tr is not None:
        axes[3].plot(*ex_acc_trace, '--', color='orange', label='Loss (train)')
        tax3 = axes[3].twinx()
        tax3.twinx().plot(*get_trace('acceleration_iter', 'train'), '--', color='black', label='Iters (train)')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Accelerated example loss')
        tax3.set_ylabel('Acceleration iters')
        axes[3].set_title('Example acceleration metrics over time')
    
    plt.tight_layout()
    fig.savefig(os.path.join(figs_dir, 'traces.jpg'), dpi=50)

def generate_animation(trial_dir):
    figs_dir = os.path.join(trial_dir, 'figures')
    os.makedirs(figs_dir, exist_ok=True)
    frames_dir = os.path.join(trial_dir, 'eg_frames')
    frames_files = os.listdir(frames_dir)
    sorted_indices = np.argsort([int(f.split('.')[0].split('_')[-1]) for f in frames_files])
    frames_files = [frames_files[idx] for idx in sorted_indices]
    with imageio.get_writer(os.path.join(figs_dir, 'images_over_time.gif'), mode='I', duration=10/len(frames_files)) as writer:
        for frame_file in frames_files:
            image = imageio.imread(os.path.join(frames_dir, frame_file))
            writer.append_data(image)
    
if __name__ == '__main__':
    for dataset, save_dir in zip([WatermarkedMNIST, ColoredMNIST], ['gan_trial__watermarked', 'gan_trial__colored']):
        if '--run-trial' in sys.argv:
            run_trial(dataset=dataset, save_dir=save_dir)
        if '--generate-figs' in sys.argv:
            try:
                plot_traces(save_dir)
            except BaseException as e:
                print('Failed to plot traces. Exception: {}'.format(e))
            try:
                generate_animation(save_dir)
            except BaseException as e:
                print('Failed to animate images. Exception: {}'.format(e))