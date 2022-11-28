import numpy as np
import torch
from torch import nn

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

def execute_epoch(execute_fn, dataloader, *args, callback=None, **kwargs):
    Results = {}
    for batch in dataloader:
        results = execute_fn(batch, *args, **kwargs)
        if callback != None:
            callback()
        for key in results.keys():
            if not key in Results.keys():
                Results[key] = []
            Results[key].append(results[key])
    return Results

def calculate_auc(dataset, model, device, batch_size=1):
    rank_over_time = {}
    for label in dataset.classes:
        traces = dataset.get_traces_for_label(label)
        indices = np.arange(len(traces))
        np.random.shuffle(indices)
        batches = [
            torch.stack(traces[batch_size*i:batch_size*(i+1)]).to(device)
            for i in range(int(np.ceil(len(traces)/batch_size)))]
        output_dists = torch.cat([nn.functional.softmax(model(batch), dim=-1) for batch in batches])
        mean_ranks = [
            mean_rank(torch.sum(torch.log(output_dists[:i])), label)
            for i in range(1, len(output_dists)+1)]
        rank_over_time[label] = mean_ranks
    auc = np.mean([np.sum(rot) for _, rot in rank_over_time.items()])
    return auc