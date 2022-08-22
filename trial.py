from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from model_trainer import GanTrainer

def run_trial(g_model, d_model, g_model_kwargs, d_model_kwargs,
              g_loss_fn, d_loss_fn, g_loss_fn_kwargs, d_loss_fn_kwargs,
              g_optimizer, d_optimizer, g_optimizer_kwargs, d_optimizer_kwargs,
              train_dataset, train_dataloader_kwargs, val_dataset, val_dataloader_kwargs,
              device, trial_kwargs):
    generator_model = g_model(**g_model_kwargs)
    discriminator_model = d_model(**d_model_kwargs)
    generator_loss_fn = g_loss_fn(**g_loss_fn_kwargs)
    discriminator_loss_fn = d_loss_fn(**d_loss_fn_kwargs)
    generator_optimizer = g_optimizer(generator_model.parameters(), **g_optimizer_kwargs)
    discriminator_optimizer = d_optimizer(discriminator_model.parameters(), **d_optimizer_kwargs)
    gan_trainer = GanTrainer(generator_model, discriminator_model,
                             generator_loss_fn, discriminator_loss_fn,
                             generator_optimizer, discriminator_optimizer,
                             device)
    train_dataloader = DataLoader(train_dataset, **train_dataloader_kwargs)
    if val_dataset != None:
        val_dataloader = DataLoader(val_dataset, **val_dataloader_kwargs)
    else:
        val_dataloader = None
    
    # Lists to store the generator/discriminator training/validation loss over time
    g_losses_train, d_losses_train, obfuscated_signals_train, predictions_train, g_losses_val, d_losses_val, obfuscated_signals_val, predictions_val = [], [], [], [], [], [], [], []
    # Function to record the current 
    def evaluate_losses():
        d_loss_train, g_loss_train, obfuscated_signal_train, prediction_train = gan_trainer.eval_epoch(train_dataloader)
        d_losses_train.append(np.mean(d_loss_train))
        g_losses_train.append(np.mean(g_loss_train))
        obfuscated_signals_train.append(obfuscated_signal_train)
        predictions_train.append(prediction_train)
        if val_dataloader != None:
            d_loss_val, g_loss_val, obfuscated_signal_val, prediction_val = gan_trainer.eval_epoch(val_dataloader)
            d_losses_val.append(np.mean(d_loss_val))
            g_losses_val.append(np.mean(g_loss_val))
            obfuscated_signals_val.append(obfuscated_signal_val)
            predictions_val.append(prediction_val)
        
    # Gather initial results
    evaluate_losses()
    # Pretrain discriminator
    for epoch_idx in tqdm(range(trial_kwargs['d_pretrain_epochs'])):
        gan_trainer.train_epoch_d(train_dataloader)
        evaluate_losses()
    # Pretrain generator
    for epoch_idx in tqdm(range(trial_kwargs['g_pretrain_epochs'])):
        gan_trainer.train_epoch_g(train_dataloader)
        evaluate_losses()
    # Train GAN system
    for epoch_idx in tqdm(range(trial_kwargs['gan_train_epochs'])):
        gan_trainer.train_epoch(train_dataloader)
        evaluate_losses()
    # Posttrain discriminator
    for epoch_idx in tqdm(range(trial_kwargs['d_posttrain_epochs'])):
        gan_trainer.train_epoch_d(train_dataloader)
        evaluate_losses()
    # Posttrain generator
    for epoch_idx in tqdm(range(trial_kwargs['g_posttrain_epochs'])):
        gan_trainer.train_epoch_g(train_dataloader)
        evaluate_losses()
    
    return g_losses_train, d_losses_train, obfuscated_signals_train, predictions_train, g_losses_val, d_losses_val, obfuscated_signals_val, predictions_val