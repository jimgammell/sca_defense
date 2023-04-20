import numpy as np
import torch
from torch import nn
from tqdm import tqdm

@torch.no_grad()
def val(x):
    if not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()

@torch.no_grad()
def int_to_binary(x):
    assert x.dtype == torch.long
    out = torch.zeros(x.size(0), 8, dtype=torch.long, device=x.device)
    for n in range(7, -1, -1):
        high = torch.ge(x, 2**n)
        out[:, n] = high
        x -= high*2**n
    return out

@torch.no_grad()
def acc(x, y):
    x, y = val(x), val(y)
    return np.mean(np.equal(np.argmax(x, axis=-1), y))

@torch.no_grad()
def bin_acc(x, y):
    x, y = val(x), val(y)
    return np.mean(np.equal(x>0, y))

def train_step(batch, model, optimizer, lr_scheduler, device):
    rv = {}
    model.train()
    trace, labels = batch
    trace, labels = trace.to(device), {key: item.to(device) for key, item in labels.items()}
    logits = model(trace)
    
    loss = 0.0
    for head_name, head_logits in logits.items():
        tr, tap, tb = head_name.split('__')
        target = labels['{}__{}'.format(tap, tb)]
        if tr == 'bits':
            target = int_to_binary(target)
            loss_h = nn.functional.binary_cross_entropy(torch.sigmoid(head_logits), target.to(torch.float))
            acc_h = bin_acc(head_logits, target)
        elif tr == 'bytes':
            loss_h = nn.functional.cross_entropy(head_logits, target)
            acc_h = acc(head_logits, target)
        else:
            raise NotImplementedError
        loss += loss_h
        rv[head_name+'__loss'] = loss_h.detach().cpu().numpy()
        rv[head_name+'__acc'] = acc_h
    loss /= len(logits)
    rv['total_loss'] = val(loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    return rv

def eval_step(batch, model, device):
    rv = {}
    model.eval()
    trace, labels = batch
    trace, labels = trace.to(device), {key: item.to(device) for key, item in labels.items()}
    logits = model(trace)
    
    loss = 0.0
    for head_name, head_logits in logits.items():
        tr, tap, tb = head_name.split('__')
        target = labels['{}__{}'.format(tap, tb)]
        if tr == 'bits':
            target = int_to_binary(target)
            loss_h = nn.functional.binary_cross_entropy(torch.sigmoid(head_logits), target.to(torch.float))
            acc_h = bin_acc(head_logits, target)
        elif tr == 'bytes':
            loss_h = nn.functional.cross_entropy(head_logits, target)
            acc_h = acc(head_logits, target)
        else:
            raise NotImplementedError
        loss += loss_h
        rv[head_name+'__loss'] = loss_h.detach().cpu().numpy()
        rv[head_name+'__acc'] = acc_h
    loss /= len(logits)
    rv['total_loss'] = val(loss)
    
    return rv

def run_epoch(dataloader, step_fn, *step_args):
    rv = {}
    for batch in tqdm(dataloader):
        batch_rv = step_fn(batch, *step_args)
        for key, item in batch_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    for key, item in rv.items():
        rv[key] = np.mean(item)
    return rv

def train_epoch(dataloader, model, optimizer, lr_scheduler, device):
    return run_epoch(dataloader, train_step, model, optimizer, lr_scheduler, device)
def eval_epoch(dataloader, model, device):
    return run_epoch(dataloader, eval_step, model, device)