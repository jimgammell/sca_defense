from copy import deepcopy
import numpy as np
import torch

class Results:
    def __init__(self):
        self.data = {}
    def evaluate(method, *args, name=None):
        result = method(*args)
        if name == None:
            name = method.__name__
        if name in self.data.keys():
            self.data[name].append(result)
        else:
            self.data.update({name: [result]})
    def retrieve(self, name):
        return self.data[name]
    def __repr__(self):
        s = ''
        s += self.__class__.__name__ + '\n'
        for key in self.data.keys():
            s += '\t' + str(key) + ':\n'
            for entry in self.data[key]:
                s += '\t\t' + str(entry) + '\n'
        return s
    def extend(self, res):
        self_cp = self.deepcopy()
        for key in self_cp.data.keys():
            if key in res.data.keys():
                self_cp.data[key].extend(res.data[key])
            else:
                pass
        for key in res.data.keys():
            if key in self_cp.data.keys():
                pass
            else:
                self_cp.data.update({key: res.data[key]})
        return self_cp
    def collapse(self, collapse_fn):
        self_cp = deepcopy(self)
        for key in self_cp.data.keys():
            self_cp.data[key] = [collapse_fn(self_cp.data[key])]
        return self_cp

def loss(loss):
    res = loss.cpu().numpy()
    return res

def mean_accuracy(logits, labels):
    predictions = np.argmax(logits.cpu().numpy(), axis=-1)
    labels = labels.cpu().numpy()
    res = np.mean(np.equal(predictions, labels))
    return res