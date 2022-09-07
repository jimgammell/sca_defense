from tqdm import tqdm
from copy import copy
import numpy as np
import torch

from utils import get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))
from trials.trial_common import clamp_model_params, extract_rv_from_tensor, get_model_params_histogram


class GanExperiment:
    def get_real_labels(self, n):
        return torch.ones((n, 1), device=self.device)
    def get_fake_labels(self, n):
        return torch.zeros((n, 1), device=self.device)
    def get_conditions(self, n):
        conditions = np.random.choice(self.conditions, (n, 1))
        conditions = torch.tensor(conditions).to(self.device)
        return conditions
    def get_gen_loss(self, disc_logits_fake, n_labels_fake):
        if self.objective_formulation in ['Naive']:
            gen_loss = -self.gen_loss_fn(disc_logits_fake, self.get_fake_labels(n_labels_fake))
        elif self.objective_formulation in ['Goodfellow']:
            gen_loss = self.gen_loss_fn(disc_logits_fake, self.get_real_labels(n_labels_fake))
        elif self.objective_formulation in ['Wasserstein']:
            gen_loss = -torch.mean(disc_logits_fake)
        else:
            assert False
        return gen_loss
    def get_disc_loss(self, disc_logits_fake, disc_logits_real, n_labels_fake, n_labels_real, granular=False):
        if self.objective_formulation in ['Naive', 'Goodfellow']:
            disc_loss_fake = self.disc_loss_fn(disc_logits_fake, self.get_fake_labels(n_labels_fake))
            disc_loss_real = self.disc_loss_fn(disc_logits_real, self.get_real_labels(n_labels_real))
            disc_loss = .5*(disc_loss_fake+disc_loss_real)
        elif self.objective_formulation in ['Wasserstein']:
            disc_loss = torch.mean(disc_logits_fake) - torch.mean(disc_logits_real)
            disc_loss_fake = disc_loss_real = disc_loss
        else:
            assert False
        if granular:
            return disc_loss_fake, disc_loss_real
        else:
            return disc_loss            
    
    def __init__(self,
                 disc,
                 gen,
                 disc_loss_fn,
                 gen_loss_fn,
                 disc_opt,
                 gen_opt,
                 latent_vars_distr,
                 device,
                 conditions=[],
                 objective_formulation='Goodfellow',
                 disc_weight_clip_value=0.01):
        self.disc = disc.to(device)
        self.gen = gen.to(device)
        self.disc_loss_fn = disc_loss_fn
        self.gen_loss_fn = gen_loss_fn
        self.disc_opt = disc_opt
        self.gen_opt = gen_opt
        self.latent_vars_distr = latent_vars_distr
        self.eval_latent_vars = self.latent_vars_distr.sample()
        self.device = device
        if len(conditions) != 0:
            self.conditional_gan = True
            self.conditions = conditions
        else:
            self.conditional_gan = False
        self.objective_formulation = objective_formulation
        self.disc_weight_clip_value = disc_weight_clip_value
        
    def train_step(self, batch, train_disc=True, train_gen=True):
        self.disc.train()
        self.gen.train()
        
        real_images, real_labels = batch
        real_images = real_images.to(self.device)
        real_labels = real_labels.to(self.device)
        gen_latent_variables = self.latent_vars_distr.sample().to(self.device)
        disc_latent_variables = self.latent_vars_distr.sample().to(self.device)
        
        if train_gen:
            if self.conditional_gan:
                conditions = self.get_conditions(gen_latent_variables.size(0))
                fake_images = self.gen(gen_latent_variables, conditions)
                disc_logits_fake = self.disc(fake_images, conditions)
            else:
                fake_images = self.gen(gen_latent_variables)
                disc_logits_fake = self.disc(fake_images)
            gen_loss = self.get_gen_loss(disc_logits_fake, fake_images.size(0))
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
        
        if train_disc:
            if self.conditional_gan:
                conditions = self.get_conditions(disc_latent_variables.size(0))
                fake_images = self.gen(disc_latent_variables, conditions).detach()
                disc_logits_fake = self.disc(fake_images, conditions)
                disc_logits_real = self.disc(real_images, real_labels)
            else:
                fake_images = self.gen(disc_latent_variables).detach()
                disc_logits_fake = self.disc(fake_images)
                disc_logits_real = self.disc(real_images)
            disc_loss = self.get_disc_loss(disc_logits_fake, disc_logits_real, fake_images.size(0), real_images.size(0))
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            
        if self.objective_formulation in ['Wasserstein']:
            clamp_model_params(self.disc, self.disc_weight_clip_value)
    
    @torch.no_grad()
    def eval_step(self, batch):
        self.disc.eval()
        self.gen.eval()
        
        real_images, real_labels = batch
        real_images = real_images.to(self.device)
        real_labels = real_labels.to(self.device)
        latent_variables = self.latent_vars_distr.sample().to(self.device)
        
        if self.conditional_gan:
            conditions = self.get_conditions(latent_variables.size(0))
            fake_images = self.gen(latent_variables, conditions)
            disc_logits_fake = self.disc(fake_images, conditions)
            disc_logits_real = self.disc(real_images, real_labels)
        else:
            fake_images = self.gen(latent_variables)
            disc_logits_fake = self.disc(fake_images)
            disc_logits_real = self.disc(real_images)
        gen_loss = self.get_gen_loss(disc_logits_fake, fake_images.size(0))
        disc_loss_fake, disc_loss_real = self.get_disc_loss(
            disc_logits_fake, disc_logits_real, fake_images.size(0), real_images.size(0), granular=True)
        disc_preds_fake = torch.ge(disc_logits_fake, 0.5)
        disc_preds_real = torch.ge(disc_logits_real, 0.5)
        disc_acc_fake = torch.mean(torch.eq(disc_preds_fake, self.get_fake_labels(fake_images.size(0))).to(torch.float))
        disc_acc_real = torch.mean(torch.eq(disc_preds_real, self.get_real_labels(real_images.size(0))).to(torch.float))
        
        return {'gen_loss': extract_rv_from_tensor(gen_loss),
                'disc_loss_fake': extract_rv_from_tensor(disc_loss_fake),
                'disc_loss_real': extract_rv_from_tensor(disc_loss_real),
                'disc_acc_fake': extract_rv_from_tensor(disc_acc_fake),
                'disc_acc_real': extract_rv_from_tensor(disc_acc_real)}
    
    @torch.no_grad()
    def eval_disc_performance_step(self, batch):
        self.gen.eval()
        self.disc.eval()
        
        real_images, real_labels = batch
        real_images = real_images.to(self.device)
        real_labels = real_labels.to(self.device)
        latent_variables = self.latent_vars_distr.sample().to(self.device)
        
        if self.conditional_gan:
            conditions = self.get_conditions(latent_variables.size(0))
            fake_images = self.gen(latent_variables, conditions)
            disc_logits_fake = self.disc(fake_images, conditions)
            disc_logits_real = self.disc(real_images, real_labels)
        else:
            fake_images = self.gen(latent_variables)
            disc_logits_fake = self.disc(fake_images)
            disc_logits_real = self.disc(real_images)
        _, disc_loss = self.get_disc_loss(
            disc_logits_fake, disc_logits_real, fake_images.size(0), real_images.size(0), granular=True)
        disc_preds = torch.ge(disc_logits_real, 0.5)
        disc_acc = torch.mean(torch.eq(disc_preds, self.get_real_labels(real_images.size(0))).to(torch.float))
        return {'disc_loss': extract_rv_from_tensor(disc_loss),
                'disc_acc': extract_rv_from_tensor(disc_acc)}
    
    @torch.no_grad()
    def sample_generated_images(self):
        self.gen.eval()
        
        latent_variables = self.eval_latent_vars.to(self.device)
        if self.conditional_gan:
            conditions = [self.conditions[idx%len(self.conditions)]
                          for idx in range(latent_variables.size(0))]
            conditions = torch.tensor(conditions).to(self.device)
            fake_images = self.gen(latent_variables, conditions)
        else:
            fake_images = self.gen(latent_variables)
            
        return {'fake_images': extract_rv_from_tensor(fake_images)}
    
    def train_epoch(self, dataloader, train_gen=True, train_disc=True, disc_steps_per_gen_step=1):
        disc_steps = 0
        for batch in tqdm(dataloader):
            if disc_steps < disc_steps_per_gen_step-1:
                self.train_step(batch, train_gen=False, train_disc=train_disc)
                disc_steps += 1
            else:
                self.train_step(batch, train_gen=train_gen, train_disc=train_disc)
                disc_steps = 0
    
    def eval_epoch(self,
                   train_dataloader=None,
                   eval_training_metrics=True,
                   test_dataloader=None,
                   eval_disc_generalization=True,
                   eval_disc_hist=True,
                   eval_gen_hist=True,
                   sample_gen_images=True,
                   average_metrics=True):
        Results = {}
        progress_bar = tqdm(total=(len(train_dataloader) if train_dataloader!=None else 0) +\
                                  (len(test_dataloader) if test_dataloader!=None else 0) +\
                                  (1 if eval_disc_hist else 0) +\
                                  (1 if eval_gen_hist else 0) +\
                                  (1 if sample_gen_images else 0))
        if eval_training_metrics:
            assert train_dataloader != None
            Results['training_metrics'] = {}
            for batch in train_dataloader:
                results = self.eval_step(batch)
                for key in results.keys():
                    if not key in Results['training_metrics'].keys():
                        Results['training_metrics'][key] = []
                    Results['training_metrics'][key].append(results[key])
                progress_bar.update(1)
            if average_metrics:
                for key, item in Results['training_metrics'].items():
                    Results['training_metrics'][key] = np.mean(item)
        if eval_disc_generalization:
            assert test_dataloader != None
            Results['disc_generalization'] = {}
            for batch in test_dataloader:
                results = self.eval_disc_performance_step(batch)
                for key in results.keys():
                    if not key in Results['disc_generalization'].keys():
                        Results['disc_generalization'][key] = []
                    Results['disc_generalization'][key].append(results[key])
                progress_bar.update(1)
            if average_metrics:
                for key, item in Results['disc_generalization'].items():
                    Results['disc_generalization'][key] = np.mean(item)
        if eval_disc_hist:
            disc_hist = get_model_params_histogram(self.disc)
            Results['disc_hist'] = disc_hist
            progress_bar.update(1)
        if eval_gen_hist:
            gen_hist = get_model_params_histogram(self.gen)
            Results['gen_hist'] = gen_hist
            progress_bar.update(1)
        if sample_gen_images:
            sampled_images = self.sample_generated_images()
            Results['sampled_gen_images'] = sampled_images
            progress_bar.update(1)
        return Results