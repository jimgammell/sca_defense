import numpy as np
import torch
from torch import nn

class GanTrainer:
    def __init__(self,
                 generator_model, discriminator_model,
                 generator_loss_fn, discriminator_loss_fn,
                 generator_optimizer, discriminator_optimizer,
                 device):
        def parse_device(device):
            if device != None:
                return device
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.generator_loss_fn = generator_loss_fn
        self.discriminator_loss_fn = discriminator_loss_fn
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.device = parse_device(device)
        self.generator_model = self.generator_model.to(self.generator_device)
        self.discriminator_model = self.discriminator_model.to(self.discriminator_device)
        
    def train_step_d(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        x = x + self.generator_model(y)
        prediction = self.discriminator_model(x)
        loss = self.discriminator_loss_fn(prediction, y)
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()
    
    def train_step_g(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        x = x + self.generator_model(y)
        prediction = self.discriminator_model(x)
        loss = self.generator_loss_fn(prediction, y)
        self.generator_optimizer.zero_grad()
        loss.backward()
        self.generator_optimizer.step()
    
    @torch.no_grad()
    def eval_step(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        x = x + self.generator_model(y)
        prediction = self.discriminator_model(x)
        loss_g = self.generator_loss_fn(prediction, y).cpu().numpy()
        loss_d = self.discriminator_loss_fn(prediction, y).cpu().numpy()
        return loss_g, loss_d
    
    def train_epoch_d(self, dataloader):
        for batch in dataloader:
            self.train_step_d(batch)
    
    def train_epoch_g(self, dataloader):
        for batch in dataloader:
            self.train_step_g(batch)
    
    def train_epoch(self, dataloader):
        for batch in dataloader:
            self.train_step_d(batch)
            self.train_step_g(batch)
    
    def eval_epoch(self, dataloader):
        losses_d, losses_g = [], []
        for batch in dataloader:
            loss_d, loss_g = self.eval_step(batch)
            losses_d.append(loss_d)
            losses_g.append(loss_g)
        return losses_d, losses_g