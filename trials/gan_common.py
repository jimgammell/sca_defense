import numpy as np
import torch

def train_step_discriminator(batch,
                             discriminator,
                             generator,
                             loss_fn,
                             optimizer,
                             device):
    discriminator.train()
    generator.train()
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    protected_x = x + generator(x)
    logits = discriminator(protected_x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_step_generator(batch,
                         discriminator,
                         generator,
                         loss_fn,
                         optimizer,
                         device):
    discriminator.train()
    generator.train()
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    protected_x = x + generator(x)
    logits = discriminator(protected_x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
@torch.no_grad()
def eval_step(batch,
              discriminator,
              generator,
              discriminator_loss_fn,
              generator_loss_fn,
              device):
    discriminator.eval()
    generator.eval()
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    protected_x = x + generator(x)
    logits = discriminator(protected_x)
    discriminator_loss = discriminator_loss_fn(logits, y)
    generator_loss = generator_loss_fn(logits, y)
    return {'logits': logits.cpu().numpy(),
            'discriminator_loss': discriminator_loss.cpu().numpy(),
            'generator_loss': generator_loss.cpu().numpy(),
            'protected_trace': protected_x.cpu().numpy(),
            'target': y.cpu().numpy()}

def train_epoch(dataloader,
                discriminator,
                generator,
                discriminator_loss_fn,
                generator_loss_fn,
                discriminator_optimizer,
                generator_optimizer,
                device):
    for batch in dataloader:
        train_step_discriminator(batch,
                                 discriminator,
                                 generator,
                                 discriminator_loss_fn,
                                 discriminator_optimizer,
                                 device)
        train_step_generator(batch,
                             discriminator,
                             generator,
                             generator_loss_fn,
                             generator_optimizer,
                             device)

def eval_epoch(dataloader,
               discriminator,
               generator,
               discriminator_loss_fn,
               generator_loss_fn,
               device):
    results = {}
    for batch in dataloader:
        results_val = eval_step(batch,
                                discriminator,
                                generator,
                                discriminator_loss_fn,
                                generator_loss_fn,
                                device)
        for key in results_val.keys():
            if not key in results.keys():
                results.update({key: []})
            results[key].append(results_val[key])
    return results