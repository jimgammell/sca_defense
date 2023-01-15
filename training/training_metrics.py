import numpy as np
from training.common import detach_result

def get_accuracy(_, labels, logits):
    logits = detach_result(logits)
    labels = detach_result(labels)
    predictions = np.argmax(logits, axis=-1)
    correctness = np.equal(predictions, labels)
    accuracy = np.mean(correctness)
    return accuracy

def get_mean_rank(_, labels, logits):
    logits = detach_result(logits)
    labels = detach_result(labels)
    ranks = np.array([np.count_nonzero(logits[idx]>=logits[idx][label])
                      for idx, label in enumerate(labels)])
    mean_rank = np.mean(ranks)
    return mean_rank

def get_confusion_matrix(_, labels, logits):
    logits = detach_result(logits)
    labels = detach_result(labels)
    confusion_matrix = np.zeros((logits.shape[-1], logits.shape[-1]))
    predictions = np.argmax(logits, axis=-1)
    for prediction, label in zip(predictions, labels):
        confusion_matrix[prediction, label] += 1
    confusion_matrix /= logits.shape[0]/logits.shape[-1]
    return confusion_matrix