import numpy as np
import torch
from torch import nn

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
def get_rank_over_time(dataloader, model, repetitions=1, **kwargs):
    rank_over_time = np.zeros((repetitions, dataloader.dataset.num_examples))
    device = next(model.parameters()).device
    key = dataloader.dataset.key
    for repetition_idx in range(repetitions):
        logits = []
        for batch in dataloader:
            traces, _, plaintexts = batch
            traces = traces.to(device)
            logits_for_batch = model(traces).cpu()#.numpy()
            logits_for_batch = [dataloader.dataset.reorder_logits(l.unsqueeze(0), pt).squeeze() for l, pt in zip(logits_for_batch, plaintexts)]
            logits.extend(logits_for_batch)
        predictions = [nn.functional.softmax(l, dim=-1) for l in logits]
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
    auc_mean = np.mean(area_under_curve)
    auc_std = np.std(area_under_curve)
    return auc_mean, auc_std#rot_mean, rot_std, auc_mean, auc_std