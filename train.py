from tqdm import tqdm
import numpy as np
import torch
from utils import log_print as print
from results import *

def train_gan_epoch(*args):
    dataloader, generator, discriminator, generator_loss_fn, discriminator_loss_fn, generator_optimizer, discriminator_optimizer, device = args
    
    results = Results()
    for batch in dataloader:
        key_idx, trace, plaintext, key = batch
        target = key_idx.to(device)
        trace = trace.to(device)
        plaintext = plaintext.to(device)
        key = key.to(device)
        generator_input = (key_idx, trace, plaintext, key)
        discriminator_target = (target)
        
        res = train_gan_step(generator_input, discriminator_target, generator, discriminator, generator_loss_fn, discriminator_loss_fn, generator_optimizer, discriminator_optimizer)
        
        generator_logits, discriminator_logits, generator_loss, discriminator_loss = res
        results.evaluate(loss, generator_loss, name='gan_gen_train_loss')
        results.evaluate(loss, discriminator_loss, name='gan_disc_train_loss')
        results.evaluate(mean_accuracy, discriminator_logits, target, name='gan_train_acc')
    return results

def eval_gan_epoch(*args):
    dataloader, generator, discriminator, generator_loss_fn, discriminator_loss_fn, device = args
    
    results = Results()
    for batch in dataloader:
        key_idx, trace, plaintext, key = batch
        target = key_idx.to(device)
        trace = trace.to(device)
        plaintext = plaintext.to(device)
        key = key.to(device)
        generator_input = (key_idx, trace, plaintext, key)
        discriminator_target = (target)
        
        res = eval_gan_step(generator_input, discriminator_target, generator, discriminator, generator_loss_fn, discriminator_loss_fn)
        
        generator_logits, discriminator_logits, discriminator_loss, generator_loss = res
        results.evaluate(loss, generator_loss, name='gan_gen_val_loss')
        results.evaluate(loss, discriminator_loss, name='gan_disc_val_loss')
        results.evaluate(mean_accuracy, discriminator_logits, target, name='gan_val_acc')
    return results

def train_gan_step(*args):
    generator_input, discriminator_target, generator, discriminator, generator_loss_fn, discriminator_loss_fn, generator_optimizer, discriminator_optimizer = args
    
    generator_optimizer.zero_grad()
    generator_logits = generator(*generator_input)
    discriminator_logits = discriminator(generator_logits)
    generator_loss = generator_loss_fn(discriminator_logits, discriminator_target)
    generator_loss.backward()
    generator_optimizer.step()
    
    discriminator_optimizer.zero_grad()
    discriminator_logits = discriminator(generator_logits.detach())
    discriminator_loss = discriminator_loss_fn(discriminator_logits, discriminator_target)
    discriminator_loss.backward()
    discriminator_optimizer.step()
    
    res = (generator_logits.detach(), discriminator_logits.detach(), generator_loss.detach(), discriminator_loss.detach())
    return res

def eval_gan_step(*args):
    generator_input, discriminator_target, generator, discriminator, generator_loss_fn, discriminator_loss_fn = args
    
    with torch.no_grad():
        generator_logits = generator(*generator_input)
        discriminator_logits = discriminator(generator_logits)
        generator_loss = generator_loss_fn(discriminator_logits, discriminator_target)
        discriminator_loss = discriminator_loss_fn(discriminator_logits, discriminator_target)
    
    res = (generator_logits.detach(), discriminator_logits.detach(), generator_loss.detach(), discriminator_loss.detach())
    return res

def train_single_model_epoch(*args):
    pass

def eval_single_model_epoch(*args):
    pass

def train_single_model_step(*args):
    input, target, model, loss_fn, optimizer = args
    
    optimizer.zero_grad()
    logits = model(*input)
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()
    
    res = (logits.detach(), loss.detach())
    return res

def eval_single_model_step(*args):
    input, target, model, loss_fn = args
    
    with torch.no_grad():
        logits = model(*input)
        loss = loss_fn(logits, target)
    
    res = (logits.detach(), loss.detach())
    return res