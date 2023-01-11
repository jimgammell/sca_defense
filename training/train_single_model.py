import torch
from training.common import unpack_batch, detach_result, run_epoch

def get_results(lambdas, losses, x, y, logits, metric_fns):
    if metric_fns is None:
        metric_fns = {}
    results = {
        'losses': losses,
        'total_loss': sum(lbd*loss for lbd, loss in zip(lambdas, losses)),
        **{metric_name: metric_fn(x, y, logits) for metric_name, metric_fn in metric_fns.items()}
    }
    return results

def train_step(batch, model, loss_fns, optimizer, device,
               lambdas=None, grad_clip=False, metric_fns=None):
    x, y = unpack_batch(batch, device)
    model.train()
    logits = model(x)
    if not hasattr(loss_fns, '__iter__'):
        loss_fns = [loss_fns]
    if lambdas is None:
        lambdas = len(loss_fns)*[1.0]
    losses = []
    optimizer.zero_grad()
    for lbd, loss_fn in zip(lambdas, loss_fns):
        loss = lbd*loss_fn(logits, x, y)
        loss.backward()
        losses.append(detach_result(loss))
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
    optimizer.step()
    results = get_results(lambdas, losses, x, y, logits, metric_fns)
    return results

@torch.no_grad()
def eval_step(batch, model, loss_fns, device,
              lambdas=None, metric_fns=None):
    x, y = unpack_batch(batch, device)
    model.eval()
    logits = model(x)
    if not hasattr(loss_fns, '__iter__'):
        loss_fns = [loss_fns]
    if lambdas is None:
        lambdas = len(loss_fns)*[1.0]
    losses = []
    for lbd, loss_fn in zip(lambdas, loss_fns):
        loss = lbd*loss_fn(logits, x, y)
        losses.append(detach_result(loss))
    results = get_results(lambdas, losses, x, y, logits, metric_fns)
    return results

def train_epoch(dataloader, model, loss_fns, optimizer, device,
                lambdas=None, grad_clip=False, metric_fns=None):
    results = run_epoch(train_step, dataloader, model, loss_fns, optimizer, device,
                        lambdas=lambdas, grad_clip=grad_clip, metric_fns=metric_fns)
    return results

def eval_epoch(dataloader, model, loss_fns, device,
               lambdas=None, metric_fns=None):
    results = run_epoch(eval_step, dataloader, model, loss_fns, device,
                        lambdas=lambdas, metric_fns=metric_fns)
    return results