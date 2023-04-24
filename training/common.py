import numpy as np
import torch
from torch import nn

def unpack_batch(batch, device):
    x, y, _ = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y

def detach_result(result):
    return result.detach().cpu().numpy()

def run_epoch(step_fn, dataloader, *args, **kwargs):
    results = {}
    for batch in dataloader:
        rv = step_fn(batch, *args, **kwargs)
        for key, item in rv.items():
            if not key in results.keys():
                results[key] = []
            results[key].append(item)
    return results

def val(tensor):
    try:
        return tensor.detach().cpu().numpy()
    except:
        return np.nan

def acc(logits, y):
    logits, y = val(logits), val(y)
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(np.equal(predictions, y))
    return acc

def hinge_acc(logits, y):
    logits = val(logits)
    predictions = np.sign(logits)
    acc = np.mean(np.equal(predictions, y*np.ones_like(predictions)))
    return acc

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
        param_norm = p.grad.detach().norm(2)
        total_norm += param_norm.item()**2
    total_norm = total_norm**0.5
    return total_norm

def hinge_loss(logits, y):
    return nn.functional.relu(1-y*logits).mean()