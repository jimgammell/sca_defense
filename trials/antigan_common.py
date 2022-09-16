from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim

from utils import get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))
from trials.trial_common import clamp_model_params, extract_rv_from_tensor, get_model_params_histogram
import models.lenet5, models.multilayer_perceptron

class AntiGanExperiment:
    def __init__(self,
                 disc,
                 gen,
                 disc_loss_fn,
                 gen_loss_fn,
                 disc_opt,
                 gen_opt,
                 device,
                 num_classes=256,
                 latent_vars_distr=None,
                 use_labels=False,
                 objective_formulation='Complement',
                 gen_weight_clamp=None,
                 disc_weight_clamp=None):
        self.eval_batch = None
        self.disc = disc.to(device)
        self.gen = gen.to(device)
        self.disc_loss_fn = disc_loss_fn
        self.gen_loss_fn = gen_loss_fn
        self.disc_opt = disc_opt
        self.gen_opt = gen_opt
        self.device = device
        self.num_classes = num_classes
        self.latent_vars_distr = latent_vars_distr
        if self.latent_vars_distr != None:
            self.eval_latent_variables = self.latent_vars_distr.sample()
            self.use_latent_variables = True
        self.use_labels = use_labels
        self.objective_formulation = objective_formulation
        self.gen_weight_clamp = gen_weight_clamp
        self.disc_weight_clamp = disc_weight_clamp
    
    def get_one_hot_labels(self, labels):
        oh_labels = torch.rand((labels.size(0), self.num_classes), dtype=torch.float, device=self.device)
        oh_labels = .05 + .25*oh_labels
        oh_labels += .65*nn.functional.one_hot(labels, num_classes=self.num_classes).to(torch.float).to(self.device)
        return oh_labels
    
    def get_complement_labels(self, labels):
        oh_labels = self.get_one_hot_labels(labels)
        comp_labels = 1. - oh_labels
        return comp_labels
    
    def get_gen_loss(self, disc_logits, labels, generated_image=None):
        if self.objective_formulation in ['Naive']:
            gen_loss = -self.gen_loss_fn(nn.functional.softmax(disc_logits, dim=-1), self.get_one_hot_labels(labels))
        elif self.objective_formulation in ['Complement']:
            gen_loss = self.gen_loss_fn(nn.functional.softmax(disc_logits, dim=-1), self.get_complement_labels(labels))
        elif self.objective_formulation in ['Wasserstein']:
            mean_incorrect_logits = torch.mean(disc_logits*self.get_complement_labels(labels))
            gen_loss = -mean_incorrect_logits
        elif self.objective_formulation in ['SimReduction']:
            gen_loss = self.gen_loss_fn(nn.functional.softmax(disc_logits, dim=1),
                                        nn.functional.softmax(torch.zeros_like(disc_logits), dim=1))
        elif self.objective_formulation in ['Misdirection_Max']:
            logits_rng = torch.max(disc_logits, dim=1)[0] - torch.min(disc_logits, dim=1)[0]
            logits_adj = disc_logits - logits_rng[:, None]*self.get_one_hot_labels(labels)
            incorrect_labels = torch.argmax(logits_adj, dim=1)
            incorrect_labels = nn.functional.one_hot(incorrect_labels, num_classes=self.num_classes).to(torch.float).to(self.device)
            gen_loss = self.gen_loss_fn(nn.functional.softmax(disc_logits, dim=-1), incorrect_labels)
        elif self.objective_formulation in ['Misdirection_Min']:
            logits_rng = torch.max(disc_logits, dim=1)[0] - torch.min(disc_logits, dim=1)[0]
            logits_adj = disc_logits + logits_rng[:, None]*self.get_one_hot_labels(labels)
            incorrect_labels = torch.argmin(logits_adj, dim=1)
            incorrect_labels = nn.functional.one_hot(incorrect_labels, num_classes=self.num_classes).to(torch.float).to(self.device)
            gen_loss = self.gen_loss_fn(nn.functional.softmax(disc_logits, dim=-1), incorrect_labels)
        elif self.objective_formulation in ['Randomize']:
            random_labels = torch.randint(0, self.num_classes, labels.shape)
            random_labels = nn.functional.one_hot(random_labels, num_classes=self.num_classes).to(torch.float).to(self.device)
            gen_loss = self.gen_loss_fn(nn.functional.softmax(disc_logits, dim=-1), random_labels)
        elif self.objective_formulation in ['Cancellation']:
            assert generated_image is not None
            gen_loss = self.gen_loss_fn(generated_image, torch.zeros_like(generated_image))
        else:
            assert False
        return gen_loss
    
    def get_disc_loss(self, disc_logits, labels):
        if self.objective_formulation in ['Wasserstein']:
            mean_correct_logits = torch.mean(disc_logits*self.get_one_hot_labels(labels))
            mean_incorrect_logits = torch.mean(disc_logits*self.get_complement_labels(labels))
            disc_loss = mean_incorrect_logits - mean_correct_logits
        else:
            disc_loss = self.disc_loss_fn(nn.functional.softmax(disc_logits, dim=-1), self.get_one_hot_labels(labels))
        return disc_loss
    
    def get_protected_images(self, raw_images, raw_labels):
        gen_args = []
        if self.use_latent_variables:
            latent_variables = self.latent_vars_distr.sample().to(self.device)
            gen_args.append(latent_variables)
        if self.use_labels:
            gen_args.append(raw_labels)
        protective_noise = self.gen(*gen_args)
        protected_images = torch.tanh(raw_images + 2*protective_noise)
        return protected_images
    
    def train_step(self, batch, disc_objects=None, train_disc=True, train_gen=True):
        if disc_objects == None:
            disc = self.disc
            disc_opt = self.disc_opt
            disc_loss_fn = self.get_disc_loss
            disc_weight_clamp = self.disc_weight_clamp
        else:
            disc = disc_objects['model']
            disc_opt = disc_objects['opt']
            disc_loss_fn = disc_objects['loss_fn']
            disc_weight_clamp = disc_objects['weight_clamp'] if 'weight_clamp' in disc_objects.keys() else None
            
        disc.train()
        self.gen.train()
        
        raw_images, raw_labels = batch
        raw_images = raw_images.to(self.device)
        raw_labels = raw_labels.to(self.device)
        
        if train_gen:
            protected_images = self.get_protected_images(raw_images, raw_labels)
            disc_logits = disc(protected_images)
            gen_loss = self.get_gen_loss(disc_logits, raw_labels, protected_images)
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            if self.gen_weight_clamp != None:
                clamp_model_params(self.gen, self.gen_weight_clamp)
        
        if train_disc:
            protected_images = self.get_protected_images(raw_images, raw_labels)
            disc_logits = disc(protected_images)
            disc_loss = disc_loss_fn(disc_logits, raw_labels)
            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()
            if disc_weight_clamp != None:
                clamp_model_params(disc, disc_weight_clamp)
        
    @torch.no_grad()
    def eval_step(self, batch, disc_objects=None):
        if disc_objects == None:
            disc = self.disc
            disc_loss_fn = self.get_disc_loss
        else:
            disc = disc_objects['model']
            disc_loss_fn = disc_objects['loss_fn']
        
        disc.eval()
        self.gen.eval()
        
        raw_images, raw_labels = batch
        raw_images = raw_images.to(self.device)
        raw_labels = raw_labels.to(self.device)
        
        protected_images = self.get_protected_images(raw_images, raw_labels)
        disc_logits = disc(protected_images)
        gen_loss = self.get_gen_loss(disc_logits, raw_labels, protected_images)
        disc_loss = disc_loss_fn(disc_logits, raw_labels)
        disc_preds = torch.argmax(disc_logits, dim=-1)
        disc_acc = torch.mean(torch.eq(disc_preds, raw_labels).to(torch.float))
        labels = raw_labels.cpu().numpy()
        if isinstance(self.disc.output_transform, nn.Identity):
            disc_confidences = nn.functional.softmax(disc_logits, dim=-1).cpu().numpy()
        else:
            disc_confidences = disc_logits.cpu().numpy()
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for label, conf_vec in zip(labels, disc_confidences):
            confusion_matrix[label] += conf_vec/np.count_nonzero(labels==label)
        confusion_matrix *= (self.num_classes/raw_images.size(0))
        return {'gen_loss': extract_rv_from_tensor(gen_loss),
                'disc_loss': extract_rv_from_tensor(disc_loss),
                'disc_acc': extract_rv_from_tensor(disc_acc),
                'disc_conf_mtx': confusion_matrix}
    
    @torch.no_grad()
    def sample_generated_images(self, batch):
        self.gen.eval()
        
        raw_images, raw_labels = batch
        raw_images = raw_images.to(self.device)
        raw_labels = raw_labels.to(self.device)
        
        protected_images = self.get_protected_images(raw_images, raw_labels)
        
        return {'protected_images': extract_rv_from_tensor(protected_images),
                'labels': extract_rv_from_tensor(raw_labels)}
    
    # Based on code here: https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb
    def compute_saliency(self, batch):
        self.gen.eval()
        self.disc.eval()
        
        raw_images, raw_labels = batch
        raw_images = raw_images.to(self.device)
        raw_labels = raw_labels.to(self.device)
        
        saliencies = []
        for image, label in zip(raw_images, raw_labels):
            protected_image = self.get_protected_images(image.unsqueeze(0), label.unsqueeze(0)).detach()
            protected_image.requires_grad = True
            logits = self.disc.logits(protected_image)
            score, _ = torch.max(logits, dim=1)
            score.backward()
            saliency, _ = torch.max(torch.abs(protected_image.grad[0]), dim=0)
            saliency = saliency.unsqueeze(0)
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            saliencies.append(saliency)
        
        return {'saliency': [extract_rv_from_tensor(saliency) for saliency in saliencies],
                'labels': extract_rv_from_tensor(raw_labels)}
    
    def train_epoch(self,
                    dataloader,
                    disc_objects=None,
                    train_gen=True,
                    train_disc=True,
                    disc_steps_per_gen_step=None,
                    gen_steps_per_disc_step=None):
        progress_bar = tqdm(total=len(dataloader))
        if disc_steps_per_gen_step != None:
            assert gen_steps_per_disc_step == None
            assert train_gen == True
            assert train_disc == True
            disc_step = 0
            for batch in dataloader:
                if disc_step < disc_steps_per_gen_step-1:
                    self.train_step(batch, train_gen=False, train_disc=True, disc_objects=disc_objects)
                    disc_step += 1
                else:
                    self.train_step(batch, train_gen=True, train_disc=True, disc_objects=disc_objects)
                    disc_step = 0
                progress_bar.update(1)
        elif gen_steps_per_disc_step != None:
            assert disc_steps_per_gen_step == None
            assert train_gen == True
            assert train_disc == True
            gen_step = 0
            for batch in dataloader:
                if gen_step < gen_steps_per_disc_step-1:
                    self.train_step(batch, train_gen=True, train_disc=False, disc_objects=disc_objects)
                    gen_step += 1
                else:
                    self.train_step(batch, train_gen=True, train_disc=True, disc_objects=disc_objects)
                    gen_step = 0
                progress_bar.update(1)
        else:
            for batch in dataloader:
                self.train_step(batch, train_gen=train_gen, train_disc=train_disc, disc_objects=disc_objects)
                progress_bar.update(1)
    
    def train_independent_discriminator(self,
                                        train_dataloader,
                                        test_dataloader,
                                        progress_bar=None,
                                        n_epochs=10,
                                        disc_model=models.lenet5.LeNet5,
                                        disc_model_kwargs={},
                                        disc_loss_fn=nn.CrossEntropyLoss,
                                        disc_opt=optim.Adam):
        eg_input, _ = next(iter(train_dataloader))
        model = disc_model(eg_input.shape, self.num_classes, **disc_model_kwargs).to(self.device)
        loss_fn = disc_loss_fn()
        optimizer = disc_opt(model.parameters())
        disc_objects = {
            'model': model,
            'loss_fn': loss_fn,
            'opt': optimizer}
        Results = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': [],
            'train_conf_mtx': [],
            'test_conf_mtx': []}
        def eval_disc():
            for k in Results.keys():
                Results[k].append([])
            for batch in train_dataloader:
                results = self.eval_step(batch, disc_objects=disc_objects)
                Results['train_loss'][-1].append(results['disc_loss'])
                Results['train_acc'][-1].append(results['disc_acc'])
                Results['train_conf_mtx'][-1].append(results['disc_conf_mtx'])
            progress_bar.update(1)
            for batch in test_dataloader:
                results = self.eval_step(batch, disc_objects=disc_objects)
                Results['test_loss'][-1].append(results['disc_loss'])
                Results['test_acc'][-1].append(results['disc_acc'])
                Results['test_conf_mtx'][-1].append(results['disc_conf_mtx'])
            progress_bar.update(1)
            for k in Results.keys():
                Results[k][-1] = np.mean(Results[k][-1], axis=0)
        eval_disc()
        for epoch_idx in range(n_epochs):
            for batch in train_dataloader:
                self.train_step(batch, disc_objects=disc_objects, train_gen=False)
            progress_bar.update(1)
            eval_disc()
        return Results
    
    def eval_epoch(self,
                   train_dataloader=None,
                   test_dataloader=None,
                   eval_disc_hist=True,
                   eval_gen_hist=True,
                   sample_gen_images=True,
                   sample_saliency=True,
                   average_metrics=True,
                   train_independent_discriminator=True,
                   ind_disc_epochs=20):
        Results = {}
        progress_bar = tqdm(total=(len(train_dataloader) if train_dataloader != None else 0) +\
                                  (len(test_dataloader) if test_dataloader != None else 0) +\
                                  (1 if eval_disc_hist else 0) +\
                                  (1 if eval_disc_hist else 0) +\
                                  (1 if sample_gen_images else 0) +\
                                  (1 if sample_saliency else 0) +\
                                  (ind_disc_epochs*3+2 if train_independent_discriminator else 0))
        if train_dataloader != None:
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
                    item = np.array(item)
                    Results['training_metrics'][key] = np.mean(item, axis=0)
        if test_dataloader != None:
            Results['test_metrics'] = {}
            for batch in test_dataloader:
                results = self.eval_step(batch)
                for key in results.keys():
                    if not key in Results['test_metrics'].keys():
                        Results['test_metrics'][key] = []
                    Results['test_metrics'][key].append(results[key])
                progress_bar.update(1)
            if average_metrics:
                for key, item in Results['test_metrics'].items():
                    Results['test_metrics'][key] = np.mean(item, axis=0)
        if eval_disc_hist:
            disc_hist = get_model_params_histogram(self.disc)
            Results['disc_hist'] = disc_hist
            progress_bar.update(1)
        if eval_gen_hist:
            gen_hist = get_model_params_histogram(self.gen)
            Results['gen_hist'] = gen_hist
            progress_bar.update(1)
        if sample_gen_images:
            if self.eval_batch == None:
                assert test_dataloader != None
                self.eval_batch = next(iter(test_dataloader))
            sampled_images = self.sample_generated_images(self.eval_batch)
            Results['sampled_gen_images'] = sampled_images
            progress_bar.update(1)
        if sample_saliency:
            if self.eval_batch == None:
                assert test_dataloader != None
                self.eval_batch = next(iter(test_dataloader))
            sampled_saliency = self.compute_saliency(self.eval_batch)
            Results['sampled_saliency'] = sampled_saliency
            progress_bar.update(1)
        if train_independent_discriminator:
            assert train_dataloader != None
            assert test_dataloader != None
            results = self.train_independent_discriminator(train_dataloader, test_dataloader, progress_bar=progress_bar, n_epochs=ind_disc_epochs)
            Results['ind_disc_metrics'] = results
            
        return Results