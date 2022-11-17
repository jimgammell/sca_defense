import numpy as np
import torch

def to_np(x):
    return x.detach().cpu().numpy()

def accuracy(logits, labels):
    predictions = np.argmax(to_np(logits), axis=-1)
    labels = to_np(labels)
    acc = np.mean(np.equal(predictions, labels))
    return acc

def mean_rank(logits, labels):
    logits = to_np(logits)
    labels = to_np(labels)
    ranks = np.array([np.count_nonzero(logits[idx]>=logits[idx][label]) for idx, label in enumerate(labels)])
    return np.mean(ranks)    

def local_avg(trace, length):
    return np.array([np.mean(trace[i*length:(i+1)*length]) for i in range(len(trace)//length)])

def unpack_batch(batch, device):
    trace, label = batch
    trace = trace.to(device)
    label = label.to(device)
    return trace, label

def execute_epoch(execute_fn, dataloader, *args, **kwargs):
    Results = {}
    for batch in dataloader:
        results = execute_fn(batch, *args, **kwargs)
        for key in results.keys():
            if not key in Results.keys():
                Results[key] = []
            Results[key].append(results[key])
    return Results