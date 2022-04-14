import numpy as np
import torch

def test_results():
    results = Results()
    def eval_fn():
        return 0
    for i in range(100):
        results.evaluate(eval_fn)
    assert eval_fn.__name__ in results.keys()
    assert len(results.data[eval_fn.__name__]) == 100
    assert all([entry == 0 for entry in results[eval_fn.__name__]])
    results.rename(eval_fn.__name__, 'test_name')
    assert not(eval_fn.__name__ in results.keys())
    assert 'test_name' in results.keys()
    assert len(results['test_name']) == 100
    assert all([entry == 0 for entry in results['test_name']])
    results.collapse(np.mean)
    assert len(results['test_name']) == 1
    assert results['test_name'][0] == 0
    new_results = Results()
    new_results.extend(results)
    assert 'test_name' in new_results.keys()
    assert len(new_results['test_name']) == 1
    assert new_results['test_name'][0] == 0
    new_results.extend(results)
    assert len(new_results['test_name']) == 2
    assert all([entry == 0 for entry in new_results['test_name']])

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
    def keys(self):
        return self.data.keys()
    def __getitem__(self, name):
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
    def collapse(self, collapse_fn, keys=None):
        if keys == None:
            keys = self.data.keys()
        for key in keys:
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

def get_traces(generator, batch, device):
    key_idx, trace, plaintext, key = batch
    trace = trace.to(device)
    plaintext = plaintext.to(device)
    key = key.to(device)
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        protective_traces = generator.get_protective_trace(key_idx, trace, plaintext, key)
        visible_traces = generator(key_idx, trace, plaintext, key)
    protective_traces = torch.unbind(protective_traces)
    visible_traces = torch.unbind(visible_traces)
    raw_traces = torch.unbind(trace)
    res = (protective_traces, raw_traces, visible_traces)
    return res

def get_saliency(discriminator, trace, device):
    trace = trace.to(device)
    discriminator.eval()
    logits = discriminator(trace)
    prediction_idx = logits.argmax(dim=-1)
    prediction = logits[:, prediction_idx]
    prediction.backward()
    saliency = prediction.grad.data.detach().numpy()
    saliency = torch.unbind(saliency)
    return saliency

def get_confusion_matrix(dataloader, generator, discriminator, device):
    generator.eval()
    discriminator.eval()
    confusion_matrix = np.zeros((16, 16))
    for batch in dataloader:
        key_idx, trace, plaintext, key = batch
        trace = trace.to(device)
        plaintext = plaintext.to(device)
        key = key.to(device)
        with torch.no_grad():
            protected_trace = generator(key_idx, trace, plaintext, key)
            logits = discriminator(protected_trace)
        predictions = logits.argmax(-1).cpu().numpy()
        for (prediction, target) in zip(predictions, key_idx):
            confusion_matrix[prediction, target] += 1
    confusion_matrix /= dataloader.batch_size*len(dataloader)
    return confusion_matrix
