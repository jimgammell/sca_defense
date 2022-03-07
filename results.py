import os
import pickle
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

def get_trace(results_dict):
    epochs = [e for e in results_dict]
    epochs.extend([epochs[-1] + .5])
    x = []
    for (e0, e1) in zip(epochs[:-1], epochs[1:]):
        if hasattr(results_dict[e0], '__len__'):
            x.extend([xx for xx in np.linspace(e0, e1 - (e1-e0)*(1/len(results_dict[e0])), len(results_dict[e0]))])
        else:
            x.append(0)
    x = np.array(x)
    y = []
    for e in epochs[:-1]:
        if hasattr(results_dict[e], '__len__'):
            y.extend(results_dict[e])
        else:
            y.append(results_dict[e])
    y = np.array(y)
    return (x, y, epochs)

def generate_loss_figure(Dloss, Gloss, key):
    (depochs, dloss, epochs) = get_trace(Dloss[key])
    (gepochs, gloss, _) = get_trace(Gloss[key])
    (fig, ax) = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(depochs, dloss, '.', markersize=.5, color='blue')
    ax[1].plot(gepochs, gloss, '.', markersize=.5, color='red')
    ax[0].set_xlabel('Step')
    ax[1].set_xlabel('Step')
    ax[0].set_ylabel('Discriminator loss')
    ax[1].set_ylabel('Generator loss')
    ax[0].set_title('Discriminator training')
    ax[1].set_title('Generator training')
    ax[0].set_yscale('symlog')
    ax[1].set_yscale('symlog')
    fig.suptitle('Performance on key %x'%(key))
    for idx in range(len(epochs[1:-1])//2):
        ds = epochs[2*idx+1]
        de = epochs[2*idx+2]
        ax[0].axvspan(ds, de, alpha=.25, color='blue')
        ax[1].axvspan(ds, de, alpha=.25, color='blue')
    for idx in range(len(epochs[2:])//2):
        gs = epochs[2*idx+2]
        ge = epochs[2*idx+3]
        ax[0].axvspan(gs, ge, alpha=.25, color='red')
        ax[1].axvspan(gs, ge, alpha=.25, color='red')
    plt.tight_layout()
    return fig

def generate_saliency_figure(Saliency, key, step):
    (original_trace, saliency) = Saliency[key][step]
    (fig, ax) = plt.subplots(1, 2)
    
    print(key, step)
    print(original_trace.shape)
    print(saliency.shape)

def display_results(results):
    out = {
        'loss_over_time': None,
        'saliency': []}
    
    return out

def save_results(results, dest):
    with open(os.path.join(os.getcwd(), dest, 'results.pickle'), 'wb') as F:
        pickle.dump(results, F)