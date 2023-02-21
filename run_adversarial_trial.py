import os
import time
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import optim, nn

def run_trial(
    batch_size=512,
    device='cuda',
    disc_pretrain_epochs=20,
    gen_pretrain_epochs=20,
    train_epochs=100,
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
    save_dir=None):
    
    if save_dir is None:
        save_dir = os.path.join('.', 'results', 'adversarially_train')
    eg_frames_dir = os.path.join(save_dir, 'eg_frames')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eg_frames_dir, exist_ok=True)
    debug_log = open(os.path.join(save_dir, 'debug_log.txt'), 'w')
    def printl(*args, **kwargs):
        print(*args, file=debug_log, **kwargs)
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
    gen = LeNet5Sanitizer(use_sn=gen_sn).to(device)
    disc_opt = optim.Adam(disc.parameters(), betas=(0.5, 0.999))
    gen_opt = optim.Adam(gen.parameters())
    disc_loss_fn = nn.CrossEntropyLoss()
    printl('Disc: {}'.format(disc))
    printl('Gen: {}'.format(gen))
    printl('Disc opt: {}'.format(disc_opt))
    printl('Gen opt: {}'.format(gen_opt))
    printl('Disc loss fn: {}'.format(disc_loss_fn))
    printl('Gen loss fn: {}'.format(gen_loss_fn))
    
    results = {'epoch': []}
    current_epoch = 0
    def update_results(pretrain_disc_phase=False, pretrain_gen_phase=False):
        assert int(pretrain_disc_phase) + int(pretrain_gen_phase) <= 1
        nonlocal results, current_epoch
        printl()
        printl('Starting epoch {}'.format(current_epoch))
        t0 = time.time()
        if current_epoch == 0:
            rv = eval_epoch(train_dataloader, disc, gen, disc_loss_fn, gen_loss_fn, device,
                            pretrain_disc_phase=False,
                            pretrain_gen_phase=False,
                            ce_kwargs={'eps': ce_eps, 'warmup_iter': ce_warmup_iter, 'max_iter': ce_max_iter,
                                       'opt_const': ce_opt, 'opt_kwargs': ce_opt_kwargs},
                            loss_mixture_coefficient=loss_mixture_coefficient)
        else:
            rv = train_epoch(train_dataloader, disc, gen, disc_opt, gen_opt, disc_loss_fn, gen_loss_fn, device,
                             pretrain_disc_phase=pretrain_disc_phase,
                             pretrain_gen_phase=pretrain_gen_phase,
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
                        return_example=not(pretrain_disc_phase or pretrain_gen_phase),
                        pretrain_disc_phase=pretrain_disc_phase,
                        pretrain_gen_phase=pretrain_gen_phase,
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
            if key not in ['epoch', 'te_confusing_example', 'te_generated_example']:
                printl('\t{}: {}'.format(key, item[-1]))
        if not(pretrain_disc_phase or pretrain_gen_phase):
            fig, axes = plt.subplots(8, 5, figsize=(20, 32))
            for eg, ax in zip(results['te_confusing_example'][-1].squeeze(), axes.flatten()[:20]):
                ax.imshow(eg, cmap='binary')
            for eg, ax in zip(results['te_generated_example'][-1].squeeze(), axes.flatten()[20:]):
                ax.imshow(eg, cmap='binary')
            plt.tight_layout()
            fig.savefig(os.path.join(eg_frames_dir, 'frame_{}.jpg'.format(current_epoch)))
        current_epoch += 1
        
    while current_epoch < disc_pretrain_epochs:
        update_results(pretrain_disc_phase=True)
    while current_epoch < disc_pretrain_epochs+gen_pretrain_epochs:
        update_results(pretrain_gen_phase=True)
    while current_epoch < disc_pretrain_epochs+gen_pretrain_epochs+train_epochs:
        update_results()
    
    with open(os.path.join(save_dir, 'results.pickle'), 'wb') as F:
        pickle.dump(results, F)