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
    def evaluate(self, method, *method_args, name=None, **method_kwargs):
        result = method(*method_args, **method_kwargs)
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
            if type(self.data[key][0] == float):
                rep_fn = lambda x: str(x)
            else:
                rep_fn = lambda x: x.shape
            s += str(key) + ': %s'%(', '.join([str(rep_fn(e)) for e in self.data[key]])) + '\n'
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

def run_special_evaluation_methods(methods, generator, discriminator, dataloader, device):
    kwargs = {'generator': generator,
              'discriminator': discriminator,
              'dataloader': dataloader,
              'device': device}
    results = Results()
    for method in methods:
        results.evaluate(method, **kwargs)
    return results        

def get_traces(**kwargs):
    generator = kwargs['generator']
    dataloader = kwargs['dataloader']
    device = kwargs['device']
    batch = next(iter(dataloader))
    key_idx, trace, plaintext, key = batch
    trace = trace.to(device)
    plaintext = plaintext.to(device)
    key = key.to(device)
    generator.eval()
    with torch.no_grad():
        protective_traces = generator(key_idx, trace, plaintext, key)
        visible_traces = trace + protective_traces
    get_bounds = lambda x: (np.min(x, axis=0), np.median(x, axis=0), np.max(x, axis=0))
    protective_traces = protective_traces.detach().squeeze().cpu().numpy()
    protective_trace_bounds = get_bounds(protective_traces)
    visible_traces = visible_traces.detach().squeeze().cpu().numpy()
    visible_trace_bounds = get_bounds(visible_traces)
    raw_traces = trace.detach().squeeze().cpu().numpy()
    raw_trace_bounds = get_bounds(raw_traces)
    res = (raw_trace_bounds, protective_trace_bounds, visible_trace_bounds)
    return res

def get_saliency(**kwargs):
    discriminator = kwargs['discriminator']
    generator = kwargs['generator']
    dataloader = kwargs['dataloader']
    device = kwargs['device']
    batch = next(iter(dataloader))
    key_idx, trace, plaintext, key = batch
    trace = trace.to(device)
    plaintext = plaintext.to(device)
    key = key.to(device)
    generator.eval()
    discriminator.eval()
    with torch.no_grad():
        protective_trace = generator(key_idx, trace, plaintext, key)
        discriminator_input = trace + protective_trace
    discriminator_input.requires_grad = True
    logits = discriminator(discriminator_input)
    prediction_logits, _ = torch.max(logits, dim=-1)
    saliency = []
    for (idx, p) in enumerate(torch.unbind(prediction_logits)):
        p.backward(retain_graph=True)
        s = discriminator_input.grad.data.detach().cpu().numpy()[idx]
        saliency.append(s)
    med_saliency = np.median(np.concatenate(saliency), axis=0)
    max_saliency = np.max(np.concatenate(saliency), axis=0)
    min_saliency = np.min(np.concatenate(saliency), axis=0)
    return (min_saliency, med_saliency, max_saliency)

def get_confusion_matrix(**kwargs):
    dataloader = kwargs['dataloader']
    generator = kwargs['generator']
    discriminator = kwargs['discriminator']
    device = kwargs['device']
    generator.eval()
    discriminator.eval()
    confusion_matrix = np.zeros((256, 256))
    for batch in dataloader:
        key_idx, trace, plaintext, key = batch
        trace = trace.to(device)
        plaintext = plaintext.to(device)
        key = key.to(device)
        with torch.no_grad():
            protective_trace = generator(key_idx, trace, plaintext, key)
            discriminator_input = trace + protective_trace
            logits = discriminator(discriminator_input)
        predictions = logits.argmax(-1).cpu().numpy()
        for (prediction, target) in zip(predictions, key_idx):
            confusion_matrix[prediction, target] += 1
    confusion_matrix /= dataloader.batch_size*len(dataloader)
    return confusion_matrix
