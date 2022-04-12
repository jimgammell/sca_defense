import numpy as np
import torch

class Results:
    def __init__(self):
        self.data = {}
    def evaluate(self, method, *args, name=None):
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
        for key in self.data.keys():
            s += str(key) + ': %s'%(', '.join([str(e) for e in self.data[key]])) + '\n'
        return s
    def extend(self, res):
        for key in self.data.keys():
            if key in res.data.keys():
                self.data[key].extend(res.data[key])
            else:
                pass
        for key in res.data.keys():
            if key in self.data.keys():
                pass
            else:
                self.data.update({key: res.data[key]})
    def collapse(self, collapse_fn):
        for key in self.data.keys():
            self.data[key] = [collapse_fn(self.data[key])]
    def rename(self, src, dest):
        for key in self.data.keys():
            assert key != dest
        if src in self.data.keys():
            self.data.update({dest: self.data[src]})
            self.data.pop(src)

def loss(loss):
    res = loss.cpu().numpy()
    return res

def mean_accuracy(logits, labels):
    predictions = np.argmax(logits.cpu().numpy(), axis=-1)
    labels = labels.cpu().numpy()
    res = np.mean(np.equal(predictions, labels))
    return res