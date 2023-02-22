import os
import wandb
from torch import nn, optim
from run_adversarial_trial import run_trial, plot_results

trial_number = 0

def run_wandb_hsweep(sweep_configuration, count=100):
    def run_trial_wrapper():
        global trial_number
        save_dir = os.path.join('.', 'results', 'adversarial_wandb_htune', 'trial_{}'.format(trial_number))
        wandb.init(mode='offline')
        kwargs = {
            'save_dir': save_dir,
            'batch_size': wandb.config.batch_size,
            'device': wandb.config.device,
            'disc_pretrain_epochs': wandb.config.disc_pretrain_epochs,
            'gen_pretrain_epochs': wandb.config.gen_pretrain_epochs,
            'train_epochs': wandb.config.train_epochs,
            'disc_sn': wandb.config.disc_sn,
            'gen_sn': wandb.config.gen_sn,
            'gen_loss_fn': getattr(nn, wandb.config.gen_loss_fn)(),
            'ce_eps': wandb.config.ce_eps,
            'ce_warmup_iter': wandb.config.ce_warmup_iter,
            'ce_max_iter': wandb.config.ce_max_iter,
            'ce_opt': getattr(optim, wandb.config.ce_opt),
            'ce_opt_kwargs': wandb.config.ce_opt_kwargs,
            'project_rec_updates': wandb.config.project_rec_updates,
            'loss_mixture_coefficient': wandb.config.loss_mixture_coefficient
        }
        run_trial(**kwargs)
        plot_results(save_dir)
        trial_number += 1
    
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_configuration, project='data_sanitization')
    wandb.agent(sweep_id, function=run_trial_wrapper, count=count)