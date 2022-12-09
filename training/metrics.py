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

def get_rank_over_time(dataset, model, batch_size=1, repetitions=1):
    rank_over_time = repetitions*[len(dataset.keys())*[]]
    for repetition_idx in range(repetitions):
        for key_idx, key in enumerate(dataset.keys):
            traces, plaintexts = dataset.get_traces_for_key(key)
            device = next(model.parameters().device)
            batches = [torch.stack(traces[batch_size*i:batch_size*(i+1)]) for i in range(int(np.ceil(len(traces)/batch_size)))]
            logits = torch.cat([model(batch) for batch in batches])
            logits = [dataset.reorder_logits(l, pt) for l, pt in zip(logits, plaintexts)]
            predictions = [nn.functional.softmax(l, dim=-1) for l in logits]
            mean_ranks = []
            accumulated_prediction = torch.log(predictions[0])
            correct_answer_found = False
            for prediction in predictions[1:]:
                rank = np.count_nonzero(th_to_np(accumulated_prediction) >= th_to_np(accumulated_prediction)[key])
                rank_over_time[repetition_idx][key_idx].append(rank)
                accumulated_prediction += torch.log(prediction)
    min_length = min(len(rank_over_time[i][j]) for i in range(repetitions) for j in range(len(dataset.keys)))
    rank_over_time = np.array([[rank_over_time[i][j][:min_length] for j in range(len(dataset.keys))] for i in range(repetitions)])
    rot_mean = np.mean(rank_over_time, axis=(0, 1))
    rot_std = np.std(rank_over_time, axis=(0, 1))
    area_under_curve = np.sum(rank_over_time, axis=2)
    auc_mean = np.mean(area_under_curve)
    auc_std = np.std(area_under_curve)
    return rot_mean, rot_std, auc_mean, auc_std