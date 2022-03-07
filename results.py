import os
import pickle
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

def get_trace(results_dict):
    epochs = [e for e in results_dict]
    epochs.extend([epochs[-1] + .5])
    depochs = []
    dloss = []
    gepochs = []
    gloss = []
    for (e1, e2) in zip(epochs[:-1], epochs[1:]):
        if e1 % 1 == 0:
            depochs.extend([x for x in np.linspace(e1, e2, len(results_dict[e1]))])
            dloss.extend(results_dict[e1])
        else:
            gepochs.extend([x for x in np.linspace(e1, e2, len(results_dict[e1]))])
            gloss.extend(results_dict[e1])
    depochs = np.array(depochs)
    dloss = np.array(dloss)
    gepochs = np.array(gepochs)
    gloss = np.array(gloss)
    return (depochs, dloss, gepochs, gloss)

def generate_loss_figure(Dloss, Gloss, key,
                        xticks0 = None,
                        xticks1 = None,
                        yticks0 = None,
                        yticks1 = None,
                        xlim0 = None,
                        xlim1 = None,
                        ylim0 = None,
                        ylim1 = None,
                        grid0 = False,
                        grid1 = False,
                        yscale0 = None,
                        yscale1 = None):
    (ddepochs, ddloss, dgepochs, dgloss) = get_trace(Dloss[key])
    (gdepochs, gdloss, ggepochs, ggloss) = get_trace(Gloss[key])
    (fig, ax) = plt.subplots(1, 2, figsize=(8, 4))
    if not(xticks0 is None):
        ax[0].set_xticks(xticks0)
    if not(xticks1 is None):
        ax[1].set_xticks(xticks1)
    if not(yticks0 is None):
        ax[0].set_yticks(yticks0)
    if not(yticks1 is None):
        ax[1].set_yticks(yticks1)
    if not(xlim0 is None):
        ax[0].set_xlim(*xlim0)
    if not(xlim1 is None):
        ax[1].set_xlim(*xlim1)
    if not(ylim0 is None):
        ax[0].set_ylim(*ylim0)
    if not(ylim1 is None):
        ax[1].set_ylim(*ylim1)
    if grid0:
        ax[0].grid()
    if grid1:
        ax[1].grid()
    ax[0].plot(ddepochs, ddloss, '.', markersize=.5, color='blue', label='Discriminator training')
    ax[0].plot(dgepochs, dgloss, '.', markersize=.5, color='red', label='Generator training')
    ax[1].plot(gdepochs, gdloss, '.', markersize=.5, color='blue', label='Discriminator training')
    ax[1].plot(ggepochs, ggloss, '.', markersize=.5, color='red', label='Generator training')
    ax[0].set_xlabel('Step')
    ax[1].set_xlabel('Step')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')
    ax[0].set_title('Discriminator')
    ax[1].set_title('Generator')
    if not(yscale0 is None):
        ax[0].set_yscale(yscale0)
    if not(yscale1 is None):
        ax[1].set_yscale(yscale1)
    l = ax[0].legend()
    for lh in l.legendHandles:
        lh._legmarker.set_markersize(10)
    l = ax[1].legend()
    for lh in l.legendHandles:
        lh._legmarker.set_markersize(10)
    fig.suptitle('Performance on key %x'%(key))
    plt.tight_layout()
    return fig

def generate_saliency_figure(Saliency, key, step):
    (original_trace, saliency) = Saliency[key][step]
    (fig, ax) = plt.subplots(1, 2, sharex=True, figsize=(8, 4))
    
    ax[0].plot(original_trace, '.', markersize=.5, color='blue')
    ax[0].set_xlabel('Sample')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Protected trace')
    ax[1].plot(saliency, '.', markersize=.5, color='blue')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Value')
    ax[1].set_title('Saliency after generator training')
    fig.suptitle('Key: %x  ;  Step: %d'%(key, step))
    plt.tight_layout()
    
    return fig

def display_results(results):
    out = {
        'loss_over_time': None,
        'saliency': []}
    
    return out

def save_results(results, dest):
    with open(os.path.join(os.getcwd(), dest, 'results.pickle'), 'wb') as F:
        pickle.dump(results, F)