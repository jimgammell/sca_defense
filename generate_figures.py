import os
import pickle
import re
import numpy as np
from matplotlib import pyplot as plt

def get_figsize(rows, cols):
    W = H = 4
    return (W*cols, H*rows)

def plot_scalar_trace(x, y, fig=None, ax=None, **kwargs):
    if fig is None:
        assert axes is None
        fig, ax = plt.subplots(1, 1, figsize=get_figsize(1, 1))
    ax.plot(x, y, **kwargs)
    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'])
    if 'xlabel' in kwargs.keys():
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs.keys():
        ax.set_ylabel(kwargs['ylabel'])
    return fig, ax

def load_traces(base_dir, keys=None, epochs=None, phases=None):
    def load_results_file(f):
        with open(os.path.join(base_dir, f), 'rb') as F:
            results = pickle.load(F)
        return results
    files = os.listdir(os.path.join(base_dir))
    results_files = [f for f in files if re.match('[a-z]+_res_[0-9]+.pickle', f) is not None]
    train_results_files = [f for f in results_files if 'train' in f]
    test_results_files = [f for f in results_files if 'test' in f]
    if epochs is None:
        epochs = [int(s.split('.')[0].split('_')[-1]) for s in train_results_files]
        assert epochs == [int(s.split('.')[0].split('_')[-1]) for s in test_results_files]
    if keys is None:
        keys = [k for k in load_results_file(train_results_files[0]).keys()]
        for f in train_results_files[1:]+test_results_files:
            assert keys == [k for k in load_results_file(f).keys()]
    traces = {
        'epochs': np.array(epochs),
        **{'train_'+str(key): np.array([load_results_file(f)[key] for f in train_results_files]) for key in keys},
        **{'test_'+str(key): np.array([load_results_file(f)[key] for f in test_results_files]) for key in keys}
    }
    indices = np.argsort(traces['epochs'])
    for key, trace in traces.items():
        traces[key] = trace[indices]
    return traces

def generate_figures(results_dir):
    traces = load_traces(results_dir)
    epochs = traces['epochs']
    metrics = set([('_'.join(k.split('_')[1:]) if 'train' in k or 'test' in k else k)
                   for k in traces.keys() if k != 'epoch'])
    scalar_metrics = [m for m in ['total_loss', 'accuracy', 'mean_rank'] if m in metrics]
    sc_fig, sc_axes = plt.subplots(
        1, len(scalar_metrics), figsize=get_figsize(1, len(scalar_metrics)), sharex=True)
    for ax, scalar_metric in zip(sc_axes, scalar_metrics):
        if 'train_'+scalar_metric in traces.keys():
            plot_scalar_trace(epochs, traces['train_'+scalar_metric], fig=sc_fig, ax=ax,
                              color='blue', label=scalar_metric+' (train)', linestyle='--')
        if 'test_'+scalar_metric in traces.keys():
            plot_scalar_trace(epochs, traces['test_'+scalar_metric], fig=sc_fig, ax=ax,
                              color='blue', label=scalar_metric+' (test)', linestyle='-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title(scalar_metric)
        ax.legend()
    plt.tight_layout()
    sc_fig.savefig(os.path.join(results_dir, 'scalar_metrics.pdf'))