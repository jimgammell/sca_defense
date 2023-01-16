import torch
from torch import nn
from training.common import unpack_batch, detach_result, run_epoch

def get_results(loss, x, y, logits, metric_fns):
    if metric_fns is None:
        metric_fns = {}
    results = {
        'loss': detach_result(loss),
        **{metric_name: metric_fn(x, y, logits) for metric_name, metric_fn in metric_fns.items()}
    }
    return results

def train_step(batch, model, loss_fn, optimizer, device,
               grad_clip=None, metric_fns=None):
    x, y = unpack_batch(batch, device)
    model.train()
    logits = model(x)
    loss = loss_fn(logits, x, y)
    optimizer.zero_grad()
    loss.backward()
    if type(grad_clip) == float:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
    optimizer.step()
    results = get_results(loss, x, y, logits, metric_fns)
    return results

@torch.no_grad()
def eval_step(batch, model, loss_fn, device,
              metric_fns=None):
    x, y = unpack_batch(batch, device)
    model.eval()
    logits = model(x)
    loss = loss_fn(logits, x, y)
    results = get_results(loss, x, y, logits, metric_fns)
    return results

def train_epoch(dataloader, model, loss_fn, optimizer, device,
                grad_clip=None, metric_fns=None, **kwargs):
    results = run_epoch(train_step, dataloader, model, loss_fn, optimizer, device,
                        grad_clip=grad_clip, metric_fns=metric_fns, **kwargs)
    return results

def eval_epoch(dataloader, model, loss_fn, device,
               metric_fns=None, **kwargs):
    results = run_epoch(eval_step, dataloader, model, loss_fn, device,
                        metric_fns=metric_fns, **kwargs)
    return results