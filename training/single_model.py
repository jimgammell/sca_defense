import torch
from torch import nn
from training.common import to_np, accuracy, mean_rank, local_avg, unpack_batch

def atenc_train_step(batch, model, loss_fn, optimizer, device, grad_clip=None):
    trace, _ = unpack_batch(batch, device)
    model.train()
    logits = model(trace)
    loss = loss_fn(logits, trace)
    optimizer.zero_grad()
    loss.backward()
    if grad_clip != None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
    optimizer.step()
    return {'loss': to_np(loss)}

@torch.no_grad()
def atenc_eval_step(batch, model, loss_fn, device):
    trace, _ = unpack_batch(batch, device)
    model.eval()
    logits = model(trace)
    loss = loss_fn(logits, trace)
    return {'loss': to_np(loss)}

def train_step(batch, model, loss_fn, optimizer, device, grad_clip=None, adversarial=False):
    trace, label = unpack_batch(batch, device)
    model.train()
    logits = model(trace)
    loss = loss_fn(logits, label)
    optimizer.zero_grad()
    loss.backward()
    if grad_clip != None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
    optimizer.step()
    try:
        rv =  {'loss': to_np(loss),
               'acc': accuracy(logits, label),
               'mean_rank': mean_rank(logits, label)}
    except:
        rv = {'loss': to_np(loss)}
    return rv

@torch.no_grad()
def eval_step(batch, model, loss_fn, device):
    trace, label = unpack_batch(batch, device)
    model.eval()
    logits = model(trace)
    loss = loss_fn(logits, label)
    try:
        rv =  {'loss': to_np(loss),
               'acc': accuracy(logits, label),
               'mean_rank': mean_rank(logits, label)}
    except:
        rv = {'loss': to_np(loss)}
    return rv