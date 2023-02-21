import os
import pickle
import time
import numpy as np
from tqdm import tqdm
import wandb
import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from datasets.classified_mnist import WatermarkedMNIST
from models.lenet import LeNet5Classifier, LeNet5Sanitizer, LeNet5Autoencoder
from data_sanitization_train import *

trial_idx = 0

class SanitizerLoss(nn.Module):
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder
        
    def rec_loss(self, image, target, classifier_logits, sanitizer_logits):
        #return nn.functional.l1_loss(sanitizer_logits, image)
        image_features = self.autoencoder.get_features(image).detach()
        reconstructed_features = self.autoencoder.get_features(sanitizer_logits)
        #rec_loss = nn.functional.mse_loss(image_features, reconstructed_features)
        rec_loss = nn.functional.binary_cross_entropy(
            nn.functional.softmax(reconstructed_features, dim=-1),
            nn.functional.softmax(image_features, dim=-1))
        return rec_loss
    
    def em_loss(self, image, target, classifier_logits, sanitizer_logits):
        return nn.functional.binary_cross_entropy(
            nn.functional.softmax(classifier_logits, dim=-1),
            nn.functional.softmax(torch.zeros_like(classifier_logits), dim=-1)
        )

def run_trial():
    global trial_idx
    base_dir = os.path.join('.', 'results', 'data_sanitization_htune')
    results_dir = os.path.join(base_dir, 'trial_%d'%(trial_idx))
    os.makedirs(results_dir, exist_ok=True)
    #print('Running trial {}'.format(trial_idx))
    #print('Results dir: {}'.format(results_dir))
    #print('Config: {}'.format(config))
    #print('\n\n')
    
    trial_idx += 1
    
    wandb.init()
    
    lr_c = wandb.config.lr_c
    lr_s = wandb.config.lr_s
    beta1_c = wandb.config.beta1_c
    beta1_s = wandb.config.beta1_s
    beta2_c = wandb.config.beta2_c
    beta2_s = wandb.config.beta2_s
    s_lbd = wandb.config.s_lbd
    c_pretrain_epochs = wandb.config.c_pretrain_epochs
    s_pretrain_epochs = wandb.config.s_pretrain_epochs
    train_epochs = wandb.config.train_epochs
    atenc_bottleneck_width = wandb.config.atenc_bottleneck_width
    atenc_sn = wandb.config.atenc_sn
    san_sn = wandb.config.san_sn
    class_sn = wandb.config.class_sn
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # init datasets
    mnist_loc = os.path.join('.', 'downloads', 'MNIST')
    os.makedirs(mnist_loc, exist_ok=True)
    train_dataset = WatermarkedMNIST(train=True, root=mnist_loc, download=True)
    s_train_dataset, c_train_dataset = torch.utils.data.random_split(train_dataset, 2*[len(train_dataset)//2])
    test_dataset = WatermarkedMNIST(train=False, root=mnist_loc, download=True)
    dataloader_kwargs = {'batch_size': 256, 'num_workers': 8}
    s_train_dataloader = torch.utils.data.DataLoader(s_train_dataset, **dataloader_kwargs, shuffle=True)
    c_train_dataloader = torch.utils.data.DataLoader(c_train_dataset, **dataloader_kwargs, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)
    
    atenc_path = os.path.join(base_dir, 'autoencoder_%d_%s.pth'%(atenc_bottleneck_width, atenc_sn))
    if os.path.exists(atenc_path):
        autoencoder_state_dict = torch.load(atenc_path)
        autoencoder = LeNet5Autoencoder(bottleneck_width=atenc_bottleneck_width, use_sn=atenc_sn).to(device)
        autoencoder.load_state_dict(autoencoder_state_dict)
    else:
        print('No existing autoencoder found. Training a new one.')
        autoencoder = LeNet5Autoencoder(bottleneck_width=atenc_bottleneck_width, use_sn=atenc_sn).to(device)
        a_opt = optim.Adam(autoencoder.parameters())
        a_loss_fn = nn.MSELoss()
        a_train_dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs, shuffle=True)
        res = train_autoencoder(a_train_dataloader, autoencoder, a_loss_fn, a_opt, device, n_epochs=25)
        print(res)
        torch.save(autoencoder.state_dict(), atenc_path)
    
    # init models
    classifier = LeNet5Classifier(output_classes=2, use_sn=class_sn).cuda()
    sanitizer = LeNet5Sanitizer(use_sn=san_sn).to(device)
    c_opt = optim.Adam(classifier.parameters(),
                       lr=lr_c,
                       betas=(beta1_c, beta2_c))
    s_opt = optim.Adam(sanitizer.parameters(),
                       lr=lr_s,
                       betas=(beta1_s, beta2_s))
    c_loss_fn = nn.CrossEntropyLoss()
    s_loss_fn = SanitizerLoss(autoencoder)
    
    results = {}
    t0 = time.time()
    def update_results(rv):
        nonlocal t0
        rv.update({'duration': time.time()-t0})
        t0 = time.time()
        wandb.log(rv)
        for key in rv.keys():
            if not key in results.keys():
                results[key] = []
            results[key].append(rv[key])

    train_args = [classifier, sanitizer, c_loss_fn, s_loss_fn, c_opt, s_opt, device]
    eval_args = [classifier, sanitizer, c_loss_fn, s_loss_fn, device]
    update_results(run_epoch([c_train_dataloader, s_train_dataloader], eval_step, eval_args, pref='tr_', s_lbd=s_lbd))
    update_results(run_epoch(test_dataloader, eval_step, eval_args, pref='te_', s_lbd=s_lbd))
    for epoch in range(s_pretrain_epochs):
        update_results(run_epoch([c_train_dataloader, s_train_dataloader], pretrain_s_step, train_args, pref='tr_', s_lbd=s_lbd))
        update_results(run_epoch(test_dataloader, eval_step, eval_args, pref='te_', s_lbd=s_lbd))
    for epoch in range(c_pretrain_epochs):
        update_results(run_epoch([c_train_dataloader, s_train_dataloader], pretrain_c_step, train_args, pref='tr_', s_lbd=s_lbd))
        update_results(run_epoch(test_dataloader, eval_step, eval_args, pref='te_', s_lbd=s_lbd))
    for epoch in range(train_epochs):
        update_results(run_epoch([c_train_dataloader, s_train_dataloader], train_step, train_args, pref='tr_', s_lbd=s_lbd))
        update_results(run_epoch(test_dataloader, eval_step, eval_args, pref='te_', s_lbd=s_lbd))
    
    #with open(os.path.join(results_dir, 'results.pickle'), 'wb') as F:
    #    pickle.dump(results, F)
        
    #(fig, axes) = plt.subplots(1, 2, sharex=True, figsize=(8, 4))
    #epochs = np.arange(1+c_pretrain_epochs+s_pretrain_epochs+train_epochs)
    #for key, item in results.items():
    #    ax = axes[0] if 'loss' in key else axes[1]
    #    ax.plot(epochs, item,
    #            color='red' if key.split('_')[1]=='c' else 'blue' if key.split('_')[2]=='em' else 'green',
    #            linestyle='--' if key.split('_')[0]=='tr' else '-',
    #            label=key)
    #for ax in axes.flatten():
    #    ax.legend()
    #    ax.set_xlabel('Epoch')
    #    ax.set_ylabel('Value')
    #    ax.set_xlim(0, s_pretrain_epochs+c_pretrain_epochs+train_epochs)
    #    ax.axvspan(0, s_pretrain_epochs, alpha=0.2, color='blue')
    #    ax.axvspan(s_pretrain_epochs, s_pretrain_epochs+c_pretrain_epochs, alpha=0.2, color='red')
    #    ax.axvspan(s_pretrain_epochs+c_pretrain_epochs, s_pretrain_epochs+c_pretrain_epochs+train_epochs, alpha=0.2, color='purple')
    #axes[0].set_yscale('log')
    #axes[0].set_title('Loss')
    #axes[1].set_title('Accuracy')
    #plt.tight_layout()
    #fig.savefig(os.path.join(results_dir, 'training_curves.png'))
    
    classifier.eval()
    sanitizer.eval()
    n_samples_to_visualize = 50
    sample_images = n_samples_to_visualize*[train_dataset.__getitem__(0)[0]]
    sample_indices = n_samples_to_visualize*[0]
    targets = (n_samples_to_visualize//10)*[idx for idx in range(10)]
    idx = 0
    for sidx, target in enumerate(targets):
        image = None
        while image is None:
            image_, _, metadata = test_dataset[idx]
            target_ = metadata['target']
            if target == target_:
                image = val(get_sanitized_image(sanitizer, image_.to(device).unsqueeze(0)).squeeze())
            else:
                idx += 1
        sample_images[sidx] = image
        sample_indices[sidx] = idx
    ax_size_per_image = 4
    images_per_row = 5
    (fig, axes) = plt.subplots(n_samples_to_visualize//images_per_row, images_per_row, 
                               figsize=(images_per_row*ax_size_per_image, n_samples_to_visualize*ax_size_per_image//images_per_row))
    for image, ax in zip(sample_images, axes.flatten()):
        ax.imshow(image, cmap='binary')
    fig.savefig(os.path.join(results_dir, 'eg_digits.png'))