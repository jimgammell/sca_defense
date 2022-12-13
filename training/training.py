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
    metrics.update({name: f(dataloader, model, dataloader.batch_size, **kwargs) for name, f in epoch_metric_fns.items()})
    return metrics

def train_batch_gan(batch, disc, gen, disc_loss_fn, gen_loss_fn, disc_optimizer, gen_optimizer, device,
                    disc_batch_metric_fns={}, gen_batch_metric_fns={}, disc_grad_clip_val=None, gen_grad_clip_val=None, **kwargs):
    disc.train()
    gen.train()
    traces, labels, plaintexts = batch
    traces, labels = traces.to(device), labels.to(device)
    gen_logits = gen(traces)
    disc_logits = disc(traces + gen_logits)
    gen_loss = gen_loss_fn(gen_logits, disc_logits, labels)
    gen_optimizer.zero_grad()
    gen_loss.backward()
    if gen_grad_clip_val is not None:
        nn.utils.clip_grad_norm_(gen.parameters(), max_norm=gen_grad_clip_val, norm_type=2)
    gen_optimizer.step()
    disc_logits = disc(traces + gen_logits.detach())
    disc_loss = disc_loss_fn(disc_logits, labels)
    disc_optimizer.zero_grad()
    disc_loss.backward()
    if disc_grad_clip_val is not None:
        nn.utils.clip_grad_norm_(disc.parameters(), max_norm=disc_grad_clip_val, norm_type=2)
    disc_optimizer.step()
    raw_disc_output = {'traces': traces, 'labels': labels, 'plaintexts': plaintexts, 'logits': disc_logits, 'loss': disc_loss}
    disc_metrics = {name: f(**raw_disc_output) for name, f in disc_batch_metric_fns.items()}
    raw_gen_output = {'traces': traces, 'labels': labels, 'plaintexts': plaintexts, 'logits': gen_logits, 'loss': gen_loss}
    gen_metrics = {name: f(**raw_gen_output) for name, f in gen_batch_metric_fns.items()}
    return disc_metrics, gen_metrics

def train_batch(batch, model, loss_fn, optimizer, device, batch_metric_fns={}, autoencoder=False, grad_clip_val=None, **kwargs):
    model.train()
    traces, labels, plaintexts = batch
    traces, labels = traces.to(device), labels.to(device).squeeze()
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
    traces, labels = traces.to(device), labels.to(device).squeeze()
    logits = model(traces)
    if autoencoder:
        loss = loss_fn(logits, traces)
    else:
        loss = loss_fn(logits, labels)
    raw_output = {'traces': traces, 'labels': labels, 'plaintexts': plaintexts, 'logits': logits, 'loss': loss}
    metrics = {name: f(**raw_output) for name, f in batch_metric_fns.items()}
    return metrics