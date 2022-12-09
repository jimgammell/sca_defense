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

def execute_epoch(execute_fn, dataloader, *args, callback=None, calc_accumulated_results=False, **kwargs):
    Results = {}
    for batch in dataloader:
        results = execute_fn(batch, *args, **kwargs)
        if callback != None:
            callback()
        for key in results.keys():
            if not key in Results.keys():
                Results[key] = []
            Results[key].append(results[key])
    if calc_accumulated_results:
        guessing_entropy, area_under_curve = calculate_accumulated_accuracy(dataloader.dataset,
                                                                            [arg for arg in args if isinstance(arg, nn.Module)][0],
                                                                            [arg for arg in args if isinstance(arg, str)][0],
                                                                            batch_size=dataloader.batch_size)
        Results['guessing_entropy'] = guessing_entropy
        Results['area_under_curve'] = area_under_curve
    return Results

def calculate_accumulated_accuracy(dataset, model, device, batch_size=256, repetitions=100):
    guessing_entropy = np.zeros(len(dataset.classes), repetitions)
    area_under_curve = np.zeros(len(dataset.classes), repetitions)
    for repetition_idx in range(repetitions):
        for label_idx, label in enumerate(dataset.classes):
            traces = [torch.from_numpy(t).view(1, -1).to(device).to(torch.float) for t in dataset.get_traces_for_label(label)]
            if dataset.transform is not None:
                for idx, trace in enumerate(traces):
                    traces[idx] = dataset.transform(trace)
            indices = np.arange(len(traces))
            np.random.shuffle(indices)
            traces = [trace[idx] for idx in indices]
            batches = [
                torch.stack(traces[batch_size*i:batch_size*(i+1)])
                for i in range(int(np.ceil(len(traces)/batch_size)))]
            output_dists = torch.cat([nn.functional.softmax(model(batch), dim=-1) for batch in batches])
            mean_ranks = []
            accumulated_dists = torch.log(output_dists[0])
            correct_answer_found = False
            for output_dist in output_dists[1:]:
                mean_ranks.append(np.count_nonzero(to_np(accumulated_dists) >= to_np(accumulated_dists)[label]))
                if not(correct_answer_found) and (np.argmax(to_np(accumulated_dists)) != label):
                    guessing_entropy[label_idx, repetition_idx] += 1
                else:
                    correct_answer_found = True
                accumulated_dists += torch.log(output_dist)
            mean_rank[label_idx, repetition_idx] = np.mean(mean_ranks)
    guessing_entropy = np.mean(guessing_entropy, axis=0)
    area_under_curve = np.mean(area_under_curve, axis=0)
    return guessing_entropy, area_under_curve