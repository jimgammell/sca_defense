import numpy as np
import torch
from torch import nn, optim

def val(tensor):
    return tensor.detach().cpu().numpy()

def acc(logits, labels):
    logits, labels = val(logits), val(labels)
    acc = np.mean(np.equal(np.argmax(logits, axis=-1), labels))
    return acc

def to_uint8(x):
    x = 255.0*(0.5*x+0.5)
    x = x.to(torch.uint8)
    x = x.to(torch.float)
    x = 2.0*(x/255.0)-1.0
    return x

def get_weight_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.detach().norm(2)
        total_norm += param_norm.item()**2
    total_norm = total_norm**0.5
    return total_norm

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm**0.5
    return total_norm

def hinge_loss(logits, y):
    return nn.functional.relu(1-y*logits).mean()

def confusion_loss(fake_output, real_output):
    loss = torch.abs(fake_output.mean(dim=0)-real_output.mean(dim=0))
    return loss

def compute_optimal_example(example, disc, gen, eps=1e-5, max_iter=100, opt=optim.Adam, opt_kwargs={'lr': 1e-2}):
    confusing_example = gen(example).detach()
    confusing_example.requires_grad = True
    opt = opt([confusing_example], **opt_kwargs)
    criterion = confusion_loss
    disc_output_real = disc(example).detach()
    prev_loss = np.inf
    for current_iter in range(max_iter):
        disc_output_fake = disc(confusing_example)
        loss = criterion(disc_output_fake, disc_output_real)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if torch.abs(loss.detach()-prev_loss) < eps:
            break
        prev_loss = loss.detach()
    return confusing_example.detach(), loss.detach().cpu().numpy(), current_iter

def train_step(batch, gen, gen_opt, disc, disc_opt, device,
               accelerate_examples=False, return_example=False, return_weight_norms=False, return_grad_norms=False,
               autoencoder_gen=False):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    disc.train()
    gen.train()
    rv = {}
    x_rec = gen(x)
    
    # compute discriminator gradients
    x_rec = gen(x).detach()
    matching_egs = x[:len(x)//2]
    nonmatching_egs = x[len(x)//2:]
    matching_batch = torch.cat((matching_egs, matching_egs), dim=1)
    if np.random.randint(2):
        nonmatching_batch = torch.cat((nonmatching_egs, gen(nonmatching_egs).detach()), dim=1)
    else:
        nonmatching_batch = torch.cat((gen(nonmatching_egs).detach(), nonmatching_egs), dim=1)
    #matching_batch = 0.75*matching_batch+0.25*torch.randn_like(matching_batch)
    #nonmatching_batch = 0.75*nonmatching_batch+0.25*torch.randn_like(nonmatching_batch)
    disc_pred_matching = disc(matching_batch)
    disc_pred_nonmatching = disc(nonmatching_batch)
    disc_loss_matching = hinge_loss(disc_pred_matching, 1)
    disc_loss_nonmatching = hinge_loss(disc_pred_nonmatching, -1)
    disc_loss = 0.5*disc_loss_matching + 0.5*disc_loss_nonmatching
    disc_opt.zero_grad()
    if disc_loss > 0:
        disc_loss.backward()
        if return_grad_norms:
            rv.update({'disc_grad_norm': get_grad_norm(disc)})
        disc_opt.step()
    elif return_grad_norms:
        rv.update({'disc_grad_norm': get_grad_norm(disc)})
    
    # compute generator gradients
    x_rec = gen(x)
    generator_batch = torch.cat((x, x_rec), dim=1)
    #generator_batch = 0.75*generator_batch+0.25*torch.randn_like(generator_batch)
    disc_pred = disc(generator_batch)
    gen_loss = -disc_pred.mean()
    gen_opt.zero_grad()
    gen_loss.backward()
    if return_grad_norms:
        rv.update({'gen_grad_norm': get_grad_norm(gen)})
    gen_opt.step()
    
    # return step information
    rv.update({
        'disc_loss': val(disc_loss),
        'gen_loss': val(gen_loss)
    })
    if return_example:
        rv.update({
            'orig_example': 0.5*val(to_uint8(x))+0.5,
            'rec_example': 0.5*val(to_uint8(x_rec))+0.5
        })
    if return_weight_norms:
        rv.update({
            'disc_weight_norm': get_weight_norm(disc),
            'gen_weight_norm': get_weight_norm(gen)
        })
    return rv

@torch.no_grad()
def eval_step(batch, gen, disc, device,
              return_example=False):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    gen.eval()
    disc.eval()
    x_rec = gen(x)
    generator_batch = torch.cat((x, x_rec), dim=1)
    #generator_batch = 0.75*generator_batch+0.25*torch.randn_like(generator_batch)
    disc_pred = disc(generator_batch)
    disc_loss = hinge_loss(disc_pred, -1)
    gen_loss = -disc_pred.mean()
    rv = {
        'disc_loss': val(disc_loss),
        'gen_loss': val(gen_loss)
    }
    if return_example:
        rv.update({
            'orig_example': 0.5*val(to_uint8(x))+0.5,
            'rec_example': 0.5*val(to_uint8(x_rec))+0.5
        })
    return rv

def train_epoch(dataloader, *step_args, return_example_idx=None, **step_kwargs):
    rv = {}
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx)==int else return_example_idx
    for bidx, batch in enumerate(dataloader):
        step_rv = train_step(batch, *step_args, return_example=bidx in re_indices, **step_kwargs)
        for key, item in step_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    for key, item in rv.items():
        if not key in ['orig_example', 'rec_example']:
            rv[key] = np.mean(item)
    return rv

def eval_epoch(dataloader, *step_args, return_example_idx=None, **step_kwargs):
    rv = {}
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx)==int else return_example_idx
    for bidx, batch in enumerate(dataloader):
        step_rv = eval_step(batch, *step_args, return_example=bidx in re_indices, **step_kwargs)
        for key, item in step_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    for key, item in rv.items():
        if not key in ['orig_example', 'rec_example']:
            rv[key] = np.mean(item)
    return rv