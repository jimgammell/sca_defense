import os
import time
import pickle
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from datasets.classified_mnist import WatermarkedMNIST
from models.lenet import LeNet5Classifier, LeNet5Autoencoder
from adversarial_train import *

def run_trial(
    batch_size=512,
    device='cuda',
    disc_pretrain_epochs=20,
    gen_pretrain_epochs=20,
    eval_disc_posttrain_epochs=20,
    train_epochs=20,
    disc_sn=True,
    gen_sn=False,
    gen_loss_fn=nn.MSELoss(),
    ce_eps=1e-5,
    ce_warmup_iter=100,
    ce_max_iter=1000,
    ce_opt=optim.Adam,
    ce_opt_kwargs={'lr': 1e-2},
    project_rec_updates=False,
    loss_mixture_coefficient=0.5,
    ind_eval_disc=True,
    save_dir=None):
    
    if save_dir is None:
        save_dir = os.path.join('.', 'results', 'adversarially_train')
    eg_frames_dir = os.path.join(save_dir, 'eg_frames')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eg_frames_dir, exist_ok=True)
    debug_log_path = os.path.join(save_dir, 'debug_log.txt')
    if os.path.exists(debug_log_path):
        os.remove(debug_log_path)
    def printl(*args, **kwargs):
        print(*args, **kwargs)
        debug_log = open(debug_log_path, 'a')
        print(*args, file=debug_log, **kwargs)
        debug_log.close()
    printl('Saving to {}'.format(save_dir))
    
    # prepare datasets
    mnist_loc = os.path.join('.', 'downloads', 'MNIST')
    train_dataset = WatermarkedMNIST(train=True, root=mnist_loc, download=True)
    test_dataset = WatermarkedMNIST(train=False, root=mnist_loc, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    printl('Using MNIST saved at {}'.format(mnist_loc))
    printl('Train dataset: {}'.format(train_dataset))
    printl('Test dataset: {}'.format(test_dataset))
    printl('Train dataloader: {}'.format(train_dataloader))
    printl('Test dataloader: {}'.format(test_dataloader))
    
    # prepare models
    disc = LeNet5Classifier(output_classes=2, use_sn=disc_sn).to(device)
    gen = LeNet5Autoencoder(use_sn=gen_sn).to(device)
    disc_opt = optim.Adam(disc.parameters(), betas=(0.5, 0.999))
    gen_opt = optim.Adam(gen.parameters())
    disc_loss_fn = nn.CrossEntropyLoss()
    printl('Disc: {}'.format(disc))
    printl('Gen: {}'.format(gen))
    printl('Disc opt: {}'.format(disc_opt))
    printl('Gen opt: {}'.format(gen_opt))
    printl('Disc loss fn: {}'.format(disc_loss_fn))
    printl('Gen loss fn: {}'.format(gen_loss_fn))
    if ind_eval_disc:
        eval_disc = LeNet5Classifier(output_classes=2).to(device)
        eval_disc_opt = optim.Adam(eval_disc.parameters())
        eval_disc_loss_fn = nn.CrossEntropyLoss()
        printl('Eval disc: {}'.format(eval_disc))
        printl('Eval disc opt: {}'.format(eval_disc_opt))
        printl('Eval disc loss fn: {}'.format(eval_disc_loss_fn))
        eval_disc_items = (eval_disc, eval_disc_opt, eval_disc_loss_fn)
    else:
        eval_disc_items = None
    
    results = {'epoch': []}
    starting_epoch = 1
    current_epoch = starting_epoch
    def update_results(pretrain_disc_phase=False,
                       pretrain_gen_phase=False,
                       pretrain_eval_disc_phase=False,
                       posttrain_eval_disc_phase=False):
        assert int(pretrain_disc_phase)+int(pretrain_gen_phase)+int(pretrain_eval_disc_phase)+int(posttrain_eval_disc_phase) <= 1
        if pretrain_eval_disc_phase or posttrain_eval_disc_phase:
            assert eval_disc_items is not None
        return_example = not(pretrain_disc_phase or pretrain_gen_phase or pretrain_eval_disc_phase or posttrain_eval_disc_phase)
        nonlocal results, current_epoch
        printl()
        printl('Starting epoch {}'.format(current_epoch))
        t0 = time.time()
        if current_epoch == 0:
            rv = eval_epoch(train_dataloader, disc, gen, disc_loss_fn, gen_loss_fn, device,
                            pretrain_disc_phase=False,
                            pretrain_gen_phase=False,
                            pretrain_eval_disc_phase=False,
                            posttrain_eval_disc_phase=False,
                            eval_disc_items=eval_disc_items,
                            ce_kwargs={'eps': ce_eps, 'warmup_iter': ce_warmup_iter, 'max_iter': ce_max_iter,
                                       'opt_const': ce_opt, 'opt_kwargs': ce_opt_kwargs},
                            loss_mixture_coefficient=loss_mixture_coefficient)
        else:
            rv = train_epoch(train_dataloader, disc, gen, disc_opt, gen_opt, disc_loss_fn, gen_loss_fn, device,
                             pretrain_disc_phase=pretrain_disc_phase,
                             pretrain_gen_phase=pretrain_gen_phase,
                             pretrain_eval_disc_phase=pretrain_eval_disc_phase,
                             posttrain_eval_disc_phase=posttrain_eval_disc_phase,
                             eval_disc_items=eval_disc_items,
                             ce_kwargs={'eps': ce_eps, 'warmup_iter': ce_warmup_iter, 'max_iter': ce_max_iter,
                                        'opt_const': ce_opt, 'opt_kwargs': ce_opt_kwargs},
                             loss_mixture_coefficient=loss_mixture_coefficient,
                             project_rec_updates=project_rec_updates)
        printl('Done training in {} seconds'.format(time.time()-t0))
        for key, item in rv.items():
            key = 'tr_'+key
            if not key in results.keys():
                results[key] = []
            results[key].append(item)
        t0 = time.time()
        rv = eval_epoch(test_dataloader, disc, gen, disc_loss_fn, gen_loss_fn, device,
                        return_example=return_example,
                        pretrain_disc_phase=pretrain_disc_phase,
                        pretrain_gen_phase=pretrain_gen_phase,
                        pretrain_eval_disc_phase=pretrain_eval_disc_phase,
                        posttrain_eval_disc_phase=posttrain_eval_disc_phase,
                        eval_disc_items=eval_disc_items,
                        ce_kwargs={'eps': ce_eps, 'warmup_iter': ce_warmup_iter, 'max_iter': ce_max_iter,
                                   'opt_const': ce_opt, 'opt_kwargs': ce_opt_kwargs},
                        loss_mixture_coefficient=loss_mixture_coefficient)
        printl('Done testing in {} seconds'.format(time.time()-t0))
        for key, item in rv.items():
            key = 'te_'+key
            if not key in results.keys():
                results[key] = []
            results[key].append(item)
        results['epoch'].append(current_epoch)
        printl('Results:')
        for key, item in results.items():
            if key not in ['epoch', 'tr_clean_example', 'te_clean_example', 'tr_confusing_example', 'tr_generated_example', 'te_confusing_example', 'te_generated_example']:
                printl('\t{}: {}'.format(key, item[-1]))
        if return_example:
            fig, axes = plt.subplots(4, 15, figsize=(60, 16))
            for eg, ax in zip(results['te_clean_example'][-1].squeeze(), axes[:, :5].flatten()):
                ax.imshow(eg, cmap='binary')
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
            for eg, ax in zip(results['te_confusing_example'][-1].squeeze(), axes[:, 5:10].flatten()):
                ax.imshow(eg, cmap='binary')
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
            for eg, ax in zip(results['te_generated_example'][-1].squeeze(), axes[:, 10:].flatten()):
                ax.imshow(eg, cmap='binary')
                for spine in ax.spines.values():
                    spine.set_edgecolor('blue')
            fig.suptitle('Epoch: {}'.format(current_epoch))
            plt.tight_layout()
            fig.savefig(os.path.join(eg_frames_dir, 'frame_{}.jpg'.format(current_epoch)))
        current_epoch += 1
        
    if eval_disc_items is not None:
        printl('\n\nPretraining the evaluation discriminator.')
        while current_epoch <= disc_pretrain_epochs:
            update_results(pretrain_eval_disc_phase=True)
        current_epoch = starting_epoch
    printl('\n\nPretraining the discriminator.')
    while current_epoch <= disc_pretrain_epochs:
        update_results(pretrain_disc_phase=True)
    current_epoch = starting_epoch
    printl('\n\nPretraining the generator.')
    while current_epoch <= gen_pretrain_epochs:
        update_results(pretrain_gen_phase=True)
    current_epoch = starting_epoch
    printl('\n\nTraining the system.')
    while current_epoch <= train_epochs:
        update_results()
    current_epoch = starting_epoch
    if eval_disc_items is not None and eval_disc_posttrain_epochs>0:
        printl('\n\nPosttraining the evaluation discriminator.')
        while current_epoch <= eval_disc_posttrain_epochs:
            update_results(posttrain_eval_disc_phase=True)
    
    with open(os.path.join(save_dir, 'results.pickle'), 'wb') as F:
        pickle.dump(results, F)

def generate_animation(results_dir):
    frames_dir = os.path.join(results_dir, 'eg_frames')
    frames_files = os.listdir(frames_dir)
    sorted_indices = np.argsort([int(f.split('.')[0].split('_')[-1]) for f in frames_files])
    frames_files = [frames_files[idx] for idx in sorted_indices]
    images = [Image.open(os.path.join(frames_dir, f)) for f in frames_files]
    images[0].save(os.path.join(results_dir, 'sanitized_images_over_time.gif'),
                   format='GIF', append_images=images[1:], save_all=True, duration=500, loop=0)

def plot_traces(results_dir):
    with open(os.path.join(results_dir, 'results.pickle'), 'rb') as F:
        results = pickle.load(F)
    ax_length = 6
    (fig, axes) = plt.subplots(1, 4, figsize=(4*ax_length, ax_length))
    
    dl_ax = axes[0]
    dl_tr_loss = results['tr_pt_disc_loss']+results['tr_disc_loss']
    dl_te_loss = results['te_pt_disc_loss']+results['te_disc_loss']
    dl_epochs = np.arange(len(dl_tr_loss))
    dl_ax.plot(dl_epochs, dl_tr_loss, linestyle='--', color='blue', label='loop-train')
    dl_ax.plot(dl_epochs, dl_te_loss, linestyle='-', color='blue', label='loop-test')
    if 'tr_pt_eval_disc_loss' in results.keys():
        dl_tr_eval_loss = results['tr_pt_eval_disc_loss']+results['tr_eval_disc_loss']+results['tr_po_eval_disc_loss']
        dl_te_eval_loss = results['te_pt_eval_disc_loss']+results['te_eval_disc_loss']+results['te_po_eval_disc_loss']
        dl_ax.plot(np.arange(len(dl_tr_eval_loss)), dl_tr_eval_loss, linestyle='--', color='green', label='ind-train')
        dl_ax.plot(np.arange(len(dl_te_eval_loss)), dl_te_eval_loss, linestyle='-', color='green', label='ind-test')
    dl_ax.set_xlabel('Epoch')
    dl_ax.set_ylabel('Loss')
    dl_ax.set_yscale('log')
    dl_ax.axvspan(0, len(results['tr_pt_disc_loss'])-.5, alpha=0.25, color='gray', label='pretraining')
    dl_ax.axvspan(len(dl_tr_loss)-.5, len(dl_tr_eval_loss)-1, alpha=0.25, color='red', label='posttraining')
    dl_ax.legend()
    dl_ax.set_title('Discriminator loss')
    dl_ax.set_xlim(0, len(dl_tr_eval_loss)-1)
    dl_ax.grid(True)
    
    da_ax = axes[1]
    da_tr_acc = results['tr_pt_disc_acc']+results['tr_disc_acc']
    da_te_acc = results['te_pt_disc_acc']+results['te_disc_acc']
    da_epochs = dl_epochs
    da_ax.plot(da_epochs, da_tr_acc, linestyle='--', color='blue', label='loop-train')
    da_ax.plot(da_epochs, da_te_acc, linestyle='-', color='blue', label='loop-test')
    if 'tr_pt_eval_disc_acc' in results.keys():
        da_tr_eval_acc = results['tr_pt_eval_disc_acc']+results['tr_eval_disc_acc']+results['tr_po_eval_disc_acc']
        da_te_eval_acc = results['te_pt_eval_disc_acc']+results['te_eval_disc_acc']+results['te_po_eval_disc_acc']
        da_ax.plot(np.arange(len(da_tr_eval_acc)), da_tr_eval_acc, linestyle='--', color='green', label='ind-train')
        da_ax.plot(np.arange(len(da_te_eval_acc)), da_te_eval_acc, linestyle='-', color='green', label='ind-test')
    da_ax.set_xlabel('Epoch')
    da_ax.set_ylabel('Accuracy')
    da_ax.set_ylim(0, 1)
    da_ax.axvspan(0, len(results['tr_pt_disc_acc'])-.5, alpha=0.25, color='gray', label='pretraining')
    da_ax.axvspan(len(da_tr_acc)-.5, len(da_tr_eval_acc)-1, alpha=0.25, color='red', label='posttraining')
    da_ax.legend()
    da_ax.set_title('Discriminator accuracy')
    da_ax.set_xlim(0, len(da_tr_eval_acc)-1)
    da_ax.grid(True)
    
    g_ax = axes[2]
    g_tr_rec_loss = results['tr_pt_gen_loss']+results['tr_gen_rec_loss']
    g_tr_adv_loss = results['tr_gen_adv_loss']
    g_te_rec_loss = results['te_pt_gen_loss']+results['te_gen_rec_loss']
    g_te_adv_loss = results['te_gen_adv_loss']
    g_ax.plot(np.arange(len(g_tr_rec_loss)), g_tr_rec_loss, linestyle='--', color='blue', label='rec-train')
    g_ax.plot(np.arange(len(g_tr_rec_loss)-len(g_tr_adv_loss), len(g_tr_rec_loss)), g_tr_adv_loss, linestyle='--', color='red', label='adv-train')
    g_ax.plot(np.arange(len(g_te_rec_loss)), g_te_rec_loss, linestyle='-', color='blue', label='adv-test')
    g_ax.plot(np.arange(len(g_te_rec_loss)-len(g_te_adv_loss), len(g_te_rec_loss)), g_te_adv_loss, linestyle='-', color='red', label='adv-test')
    g_ax.set_xlabel('Epoch')
    g_ax.set_ylabel('Loss')
    g_ax.set_yscale('log')
    g_ax.axvspan(0, len(results['tr_pt_gen_loss'])-.5, alpha=0.25, color='gray', label='pretraining')
    g_ax.legend()
    g_ax.set_title('Generator loss')
    g_ax.set_xlim(0, len(g_tr_rec_loss)-1)
    g_ax.grid(True)
    
    ce_ax = axes[3]
    ce_tr_loss = results['tr_confusing_example_loss']
    ce_te_loss = results['te_confusing_example_loss']
    ce_ax.plot(np.arange(len(ce_tr_loss)), ce_tr_loss, linestyle='--', color='blue', label='train')
    ce_ax.plot(np.arange(len(ce_te_loss)), ce_te_loss, linestyle='-', color='blue', label='test')
    ce_ax.set_xlabel('Epoch')
    ce_ax.set_ylabel('Loss')
    ce_ax.set_yscale('log')
    ce_ax.legend()
    ce_ax.set_title('Confusing example loss')
    ce_ax.set_xlim(0, len(ce_tr_loss)-1)
    ce_ax.grid(True)
    
    fig.suptitle('Training curves from adversarial entropy-maximization trial')
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, 'traces.jpg'))
    
def plot_results(results_dir):
    generate_animation(results_dir)
    plot_traces(results_dir)