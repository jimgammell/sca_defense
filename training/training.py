import torch
from torch import nn

def execute_epoch(execute_fn, dataloader, model, *args, epoch_metric_fns={}, **kwargs):
    metrics = {}
    for batch in dataloader:
        results = execute_fn(batch, model, *args, **kwargs)
        for key, item in results.items():
            if not(key in metrics.keys()):
                metrics[key] = []
            metrics[key].append(results[key])
    metrics.update({name: f(dataloader.dataset, model, dataloader.batch_size, **kwargs) for name, f in epoch_metric_fns.items()})
    return metrics

def train_batch(batch, model, loss_fn, optimizer, device, batch_metric_fns={}, autoencoder=False, grad_clip_val=None, **kwargs):
    model.train()
    traces, labels, plaintexts = batch
    traces, labels = traces.to(device), labels.to(device)
    logits = model(traces)
    if autoencoder:
        loss = loss_fn(logits, traces)
    else:
        loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    if grad_clip_val is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val, norm_type=2)
    optimizer.step()
    raw_output = {'traces': traces, 'labels': labels, 'plaintexts': plaintexts, 'logits': logits, 'loss': loss}
    metrics = {name: f(**raw_output) for name, f in batch_metric_fns.items()}
    return metrics

@torch.no_grad()
def eval_batch(batch, model, loss_fn, device, batch_metric_fns={}, autoencoder=False, **kwargs):
    model.eval()
    traces, labels, plaintexts = batch
    traces, labels = traces.to(device), labels.to(device)
    logits = model(traces)
    if autoencoder:
        loss = loss_fn(logits, traces)
    else:
        loss = loss_fn(logits, labels)
    raw_output = {'traces': traces, 'labels': labels, 'plaintexts': plaintexts, 'logits': logits, 'loss': loss}
    metrics = {name: f(**raw_output) for name, f in batch_metric_fns.items()}
    return metrics