import time
import os
import sys
from copy import deepcopy
import pickle
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
from torch import nn, optim
from lagrangian_train import *
from models.res_autoencoder import Encoder as ResEnc, Decoder as ResDec
from datasets.classified_mnist import WatermarkedMNIST, ColoredMNIST

def run_trial(
    dataset=ColoredMNIST,
    dataset_kwargs={},
    enc=ResEnc,
    enc_kwargs={'use_sn': False},
    enc_opt=optim.Adam,
    enc_opt_kwargs={'lr': 2e-4, 'betas': [0.0, 0.999]},
    dec=ResDec,
    dec_kwargs={'use_sn': False},
    dec_opt=optim.Adam,
    dec_opt_kwargs={'lr': 2e-4, 'betas': [0.0, 0.999]},
    lbd_opt=optim.SGD,
    lbd_opt_kwargs={'lr': 1e-1, 'momentum': 0.9},
    cls=Classifier,
    cls_kwargs={'use_sn': False},
    cls_opt=optim.Adam,
    cls_opt_kwargs={'lr': 2e-4, 'betas': [0.0, 0.999]},
    icls=IndClassifier,
    icls_kwargs={},
    icls_opt=optim.Adam,
    icls_opt_kwargs={'lr': 2e-4},
    rec_loss_fn=nn.MSELoss,
    rec_loss_fn_kwargs={},
    cls_loss_fn=nn.CrossEntropyLoss,
    cls_loss_fn_kwargs={},
    icls_loss_fn=nn.CrossEntropyLoss,
    icls_loss_fn_kwargs={},
    pretrain_epochs=10,
    epochs=250,
    posttrain_epochs=50,
    batch_size=512,
    cls_steps_per_enc_step=10.0,
    separate_cls_partition=True,
    pretrain_models=True,
    save_dir=None,
    pretrain_dir=None,
    suppress_output=False,
    trial_info=None):
    
    if save_dir is None:
        save_dir = os.path.join('.', 'results', 'lagrangian_trial__colored')
    if pretrain_dir is None:
        pretrain_dir = os.path.join(save_dir, 'pretrained_models')
    eg_frames_dir = os.path.join(save_dir, 'eg_frames')
    results_dir = os.path.join(save_dir, 'results')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eg_frames_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(pretrain_dir, exist_ok=True)
    for s in ['train', 'validation', 'test']:
        os.makedirs(os.path.join(results_dir, s), exist_ok=True)
    
    device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
    
    mnist_loc = os.path.join('.', 'downloads', 'MNIST')
    train_dataset = dataset(train=True, root=mnist_loc, download=True)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (50000, 10000))
    if separate_cls_partition:
        train_dataset, cls_dataset = torch.utils.data.random_split(train_dataset, (40000, 10000))
        cls_dataset = torch.utils.data.ConcatDataset((len(train_dataset)//len(cls_dataset))*[cls_dataset])
    test_dataset = dataset(train=False, root=mnist_loc, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    if separate_cls_partition:
        cls_dataloader = torch.utils.data.DataLoader(cls_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    
    train_args, eval_args, itrain_args, ieval_args = get_models(
        enc, enc_kwargs, dec, dec_kwargs, cls, cls_kwargs, icls, icls_kwargs,
        enc_opt, enc_opt_kwargs, dec_opt, dec_opt_kwargs, lbd_opt, lbd_opt_kwargs, cls_opt, cls_opt_kwargs, icls_opt, icls_opt_kwargs,
        cls_loss_fn, cls_loss_fn_kwargs, rec_loss_fn, rec_loss_fn_kwargs, icls_loss_fn, icls_loss_fn_kwargs, device, 
        dataset.input_shape, dataset.num_classes)
    
    results = {}
    def update_results(current_epoch, eval_only=False, pretrain_phase=False, posttrain_phase=False):
        assert int(eval_only)+int(pretrain_phase)+int(posttrain_phase) <= 1
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
        if eval_only:
            train_rv = eval_epoch(train_dataloader, *eval_args, lambda_clamp_val=0.0)
        elif pretrain_phase:
            train_rv = train_epoch(train_dataloader, *train_args, lambda_clamp_val=0.0, compute_grad_norms=True)
        elif posttrain_phase:
            train_rv = train_epoch(train_dataloader, *itrain_args, indcls=True)
        else:
            if separate_cls_partition:
                train_rv = train_epoch([train_dataloader, cls_dataloader],
                                       *train_args,
                                       compute_grad_norms=True,
                                       separate_cls_partition=True)
            else:
                train_rv = train_epoch(train_dataloader,
                                       *train_args,
                                       compute_grad_norms=True,
                                       cls_steps_per_enc_step=cls_steps_per_enc_step)
        train_rv['time'] = time.time()-t0
        if posttrain_phase:
            preface_keys(train_rv, 'indcls')
        print('Done with training phase. Results:')
        print_d(train_rv)
        
        t0 = time.time()
        if pretrain_phase:
            val_rv = eval_epoch(val_dataloader, *eval_args, lambda_clamp_val=0.0, return_example_idx=0)
        elif posttrain_phase:
            val_rv = eval_epoch(val_dataloader, *ieval_args, indcls=True)
        else:
            val_rv = eval_epoch(val_dataloader, *eval_args, return_example_idx=0)
        val_rv['time'] = time.time()-t0
        if posttrain_phase:
            preface_keys(val_rv, 'indcls')
        print('Done with validation phase. Results:')
        print_d(val_rv)
        
        t0 = time.time()
        if pretrain_phase:
            test_rv = eval_epoch(test_dataloader, *eval_args, lambda_clamp_val=0.0)
        elif posttrain_phase:
            test_rv = eval_epoch(test_dataloader, *ieval_args, indcls=True)
        else:
            test_rv = eval_epoch(test_dataloader, *eval_args)
        test_rv['time'] = time.time()-t0
        if posttrain_phase:
            preface_keys(test_rv, 'indcls')
        print('Done with test phase. Results:')
        print_d(test_rv)
        
        if not posttrain_phase:
            fig, axes = plt.subplots(4, 10, figsize=(40, 16))
            for eg, ax in zip(val_rv['orig_example'][0].squeeze(), axes[:, :5].flatten()):
                eg = eg.transpose(1, 2, 0)
                ax.imshow(eg, cmap='binary')
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
            for eg, ax in zip(val_rv['rec_example'][0].squeeze(), axes[:, 5:].flatten()):
                eg = eg.transpose(1, 2, 0)
                ax.imshow(eg, cmap='binary')
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
    
    if pretrain_models and all((
        os.path.exists(os.path.join(pretrain_dir, 'encoder.pth')),
        os.path.exists(os.path.join(pretrain_dir, 'decoder.pth')),
        os.path.exists(os.path.join(pretrain_dir, 'classifier.pth')))):
        try:
            encoder_state = torch.load(os.path.join(pretrain_dir, 'encoder.pth'))
            eval_args[0].load_state_dict(encoder_state)
            decoder_state = torch.load(os.path.join(pretrain_dir, 'decoder.pth'))
            eval_args[1].load_state_dict(decoder_state)
            classifier_state = torch.load(os.path.join(pretrain_dir, 'classifier.pth'))
            eval_args[3].load_state_dict(classifier_state)
            pretrain_epochs = 0
            print('Found pretrained models; skipping pretraining step.')
        except:
            pass
    
    update_results(0, eval_only=True)
    if pretrain_models:
        for epoch_idx in range(1, pretrain_epochs+1):
            update_results(epoch_idx, pretrain_phase=True)
    else:
        pretrain_epochs = 0
    if pretrain_models and (pretrain_epochs != 0):
        torch.save(eval_args[0].state_dict(), os.path.join(pretrain_dir, 'encoder.pth'))
        torch.save(eval_args[1].state_dict(), os.path.join(pretrain_dir, 'decoder.pth'))
        torch.save(eval_args[3].state_dict(), os.path.join(pretrain_dir, 'classifier.pth'))
    train_args[1].__init__(eval_args[0].parameters(), **enc_opt_kwargs) # Reset the optimizers since we expect the optimal training directions to significantly differ from the pretraining ones
    train_args[3].__init__(eval_args[1].parameters(), **dec_opt_kwargs)
    train_args[7].__init__(eval_args[3].parameters(), **cls_opt_kwargs)
    for epoch_idx in range(pretrain_epochs+1, pretrain_epochs+epochs+1):
        update_results(epoch_idx)
    for epoch_idx in range(pretrain_epochs+epochs+1, pretrain_epochs+epochs+posttrain_epochs+1):
        update_results(epoch_idx, posttrain_phase=True)
        
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
        sorted_indices = np.argsort(epochs)
        epochs = epochs[sorted_indices]
        vals = vals[sorted_indices]
        return epochs, vals
    ax_width = 4
    (fig, axes) = plt.subplots(1, 4, figsize=(4*ax_width, ax_width))
    
    axes[0].plot(*get_trace('lagrangian', 'train'), '--', color='orange', label='lagrangianloss-train')
    axes[0].plot(*get_trace('lagrangian', 'validation'), '-', color='orange', label='lagrangianloss-val')
    axes[0].plot(*get_trace('rec_loss', 'train'), '--', color='blue', label='reconstructionloss-train')
    axes[0].plot(*get_trace('rec_loss', 'validation'), '-', color='blue', label='reconstructionloss-val')
    axes[0].plot(*get_trace('conf_loss', 'train'), '--', color='red', label='confusionloss-train')
    axes[0].plot(*get_trace('conf_loss', 'validation'), '-', color='red', label='confusionloss-val')
    tax0 = axes[0].twinx()
    tax0.plot(*get_trace('cls_acc', 'train'), '--', color='black', label='clsacc-train')
    tax0.plot(*get_trace('cls_acc', 'validation'), '-', color='black', label='clsacc-val')
    tax0.set_ylabel('Accuracy')
    tax0.set_ylim(-0.05, 1.05)
    tax0.legend()
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model performance vs. epoch')
    axes[0].legend()
    
    axes[1].plot(*get_trace('enc_grad_norm', 'train'), '-', color='green', label='encoder')
    axes[1].plot(*get_trace('dec_grad_norm', 'train'), '-', color='blue', label='decoder')
    axes[1].plot(*get_trace('lbd_grad_norm', 'train'), '-', color='orange', label='lambda')
    axes[1].plot(*get_trace('cls_grad_norm', 'train'), '-', color='red', label='classifier')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Grad norm')
    axes[1].set_title('Model grad norm vs. epoch')
    axes[1].legend()
    
    axes[2].plot(*get_trace('lambda', 'train'), '-', color='orange')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Value')
    axes[2].set_title('Lambda over time')
    
    axes[3].plot(*get_trace('indcls_loss', 'train'), '--', color='purple', label='loss-train')
    axes[3].plot(*get_trace('indcls_loss', 'test'), '-', color='purple', label='loss-val')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Loss')
    axes[3].set_yscale('log')
    axes[3].set_title('Independent classifier performance vs. epoch')
    axes[3].legend()
    tax3 = axes[3].twinx()
    tax3.plot(*get_trace('indcls_acc', 'train'), '--', color='yellow', label='acc-train')
    tax3.plot(*get_trace('indcls_acc', 'validation'), '-', color='yellow', label='acc_validation')
    tax3.set_ylim(-0.05, 1.05)
    tax3.set_ylabel('Accuracy')
    tax3.legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(figs_dir, 'traces.jpg'))

def generate_animation(trial_dir):
    figs_dir = os.path.join(trial_dir, 'figures')
    os.makedirs(figs_dir, exist_ok=True)
    frames_dir = os.path.join(trial_dir, 'eg_frames')
    frames_files = os.listdir(frames_dir)
    sorted_indices = np.argsort([int(f.split('.')[0].split('_')[-1]) for f in frames_files])
    frames_files = [frames_files[idx] for idx in sorted_indices]
    images = [Image.open(os.path.join(frames_dir, f)) for f in frames_files]
    images[0].save(os.path.join(figs_dir, 'images_over_time.gif'),
                   format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)
    
if __name__ == '__main__':
    save_dir = os.path.join('.', 'results', 'lagrangian_trial__colored')
    if '--run-trial' in sys.argv:
        run_trial(save_dir=save_dir)
    if '--generate-figs' in sys.argv:
        plot_traces(save_dir)
        generate_animation(save_dir)