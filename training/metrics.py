import numpy as np
import torch
from torch import nn
from tqdm import tqdm

def th_to_np(x):
    return x.detach().cpu().numpy()

def get_loss(loss, **kwargs):
    return th_to_np(loss)

def get_accuracy(labels, logits, **kwargs):
    labels, logits = th_to_np(labels), th_to_np(logits)
    predictions = np.argmax(logits, axis=-1)
    correctness = np.equal(labels, predictions)
    accuracy = np.mean(correctness)
    return accuracy

def get_mean_rank(labels, logits, **kwargs):
    logits, labels = th_to_np(logits), th_to_np(labels)
    ranks = np.array([np.count_nonzero(logits[idx] >= logits[idx][label]) for idx, label in enumerate(labels)])
    mean_rank = np.mean(ranks)
    return mean_rank

@torch.no_grad()
def get_rank_over_time(dataloader, model, repetitions=100, traces_per_repetition=500, generator=None, **kwargs):
    rank_over_time = np.zeros((repetitions, traces_per_repetition))
    device = next(model.parameters()).device
    key = dataloader.dataset.key
    for repetition_idx in range(repetitions):
        logits = []
        for bidx, batch in enumerate(dataloader):
            traces, _, plaintexts = batch
            traces = traces.to(device)
            if generator is not None:
                reconstructed_traces = generator(traces)
                traces = traces - reconstructed_traces
                traces = traces - torch.mean(traces)
                traces = traces / torch.std(traces)
            logits_for_batch = model(traces).cpu()#.numpy()
            logits_for_batch = [dataloader.dataset.reorder_logits(l.unsqueeze(0), pt).squeeze() for l, pt in zip(logits_for_batch, plaintexts)]
            logits.extend(logits_for_batch)
            if (bidx+1)*len(traces) >= traces_per_repetition:
                break
        predictions = [nn.functional.softmax(l, dim=-1) for l in logits][:traces_per_repetition]
        mean_ranks = []
        accumulated_prediction = torch.log(predictions[0])
        for pred_idx, prediction in enumerate(predictions[1:]):
            rank = np.count_nonzero(th_to_np(accumulated_prediction) >= th_to_np(accumulated_prediction)[key])
            rank_over_time[repetition_idx, pred_idx] = rank
            accumulated_prediction += torch.log(prediction)
        rank = np.count_nonzero(th_to_np(accumulated_prediction) >= th_to_np(accumulated_prediction)[key])
        rank_over_time[repetition_idx, pred_idx+1] = rank
    rot_mean = np.mean(rank_over_time, axis=0)
    rot_std = np.std(rank_over_time, axis=0)
    area_under_curve = np.sum(rank_over_time, axis=1)
    return rot_mean, rot_std, area_under_curve