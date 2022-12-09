import torch

def execute_epoch(execute_fn, dataloader, model, *args, **kwargs, epoch_metric_fns={}):
    metrics = {}
    for batch in dataloader:
        results = execute_fn(batch, model, *args, **kwargs)
        for key, item in results.items():
            if not(key in metrics.keys()):
                metrics[key] = []
            metrics[key].append(results[key])
    metrics.update({name: f(dataloader.dataset, model, dataloader.batch_size, **kwargs) for name, f in epoch_metric_fns.items()})
    return metrics

def train_batch(batch, model, loss_fn, optimizer, device, batch_metric_fns={}, **kwargs):
    model.train()
    traces, labels, plaintexts, keys = batch
    traces, labels = traces.to(device), labels.to(device)
    logits = model(traces)
    loss = loss_fn(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    raw_output = {'traces': traces, 'labels': labels, 'plaintexts': plaintexts,
                  'keys': keys, 'logits': logits, 'loss': loss}
    metrics = {name: f(**raw_output) for name, f in batch_metric_fns.items()}
    return metrics

@torch.no_grad()
def eval_batch(batch, model, loss_fn, device, batch_metric_fns={}, **kwargs):
    model.eval()
    traces, labels, plaintexts, keys = batch
    traces, labels = traces.to(device), labels.to(device)
    logits = model(traces)
    loss = loss_fn(logits, labels)
    raw_output = {'traces': traces, 'labels': labels, 'plaintexts': plaintexts,
                  'keys': keys, 'logits': logits, 'loss': loss}
    metrics = {name: f(**raw_output) for name, f in batch_metric_fns.items()}
    return metrics