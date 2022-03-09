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

def generate_loss_figure(Dloss, Gloss, Accuracy, key,
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
    (daepochs, daacc, gaepochs, gaacc) = get_trace(Accuracy[key])
    (fig, ax) = plt.subplots(1, 3, figsize=(12, 4))
    """if not(xticks0 is None):
        ax[0].set_xticks(xticks0)
    if not(xticks1 is None):
        ax[1].set_xticks(xticks1)
    if not(yticks0 is None):
        ax[0].set_yticks(yticks0)
    if not(yticks1 is None):
        ax[1].set_yticks(yticks1)
    if grid0:
        ax[0].grid()
    if grid1:
        ax[1].grid()"""
    ax[0].set_xlim(ddepochs[0], ddepochs[-1])
    ax[1].set_xlim(ddepochs[0], ddepochs[-1])
    ax[0].plot(ddepochs, ddloss, '.', markersize=.5, color='blue', label='Discriminator training')
    ax[0].plot(dgepochs, dgloss, '.', markersize=.5, color='red', label='Generator training')
    ax[1].plot(gdepochs, gdloss, '.', markersize=.5, color='blue', label='Discriminator training')
    ax[1].plot(ggepochs, ggloss, '.', markersize=.5, color='red', label='Generator training')
    ax[0].set_xlabel('Step')
    ax[1].set_xlabel('Step')
    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Loss')
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[2].plot(daepochs, daacc, '.', markersize=.5, color='blue', label='Discriminator training')
    ax[2].plot(gaepochs, gaacc, '.', markersize=.5, color='red', label='Generator training')
    ax[2].set_xlabel('Step')
    ax[2].set_ylabel('Accuracy')
    ax[2].set_ylim(0, 1)
    ax[0].set_title('Discriminator')
    ax[1].set_title('Generator')
    ax[2].set_title('Discriminator')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    """if not(yscale0 is None):
        ax[0].set_yscale(yscale0)
    if not(yscale1 is None):
        ax[1].set_yscale(yscale1)
    l = ax[0].legend()
    for lh in l.legendHandles:
        lh._legmarker.set_markersize(10)
    l = ax[1].legend()
    for lh in l.legendHandles:
        lh._legmarker.set_markersize(10)"""
    fig.suptitle('Performance on key %x'%(key))
    plt.tight_layout()
    return fig

def generate_saliency_figure(Saliency, key, step):
    (original_trace, disc_saliency, gen_saliency) = Saliency[key][step]
    
    fig = plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.plot(original_trace, '.', markersize=.5, color='blue')
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.set_title('Protected trace')
    
    (fig, ax) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
    ax[0].plot(disc_saliency, '.', markersize=.5, color='blue')
    ax[0].set_xlabel('Sample')
    ax[0].set_ylabel('Saliency')
    ax[0].set_title('Saliency after discriminator training')
    ax[1].plot(gen_saliency, '.', markersize=.5, color='blue')
    ax[1].set_xlabel('Sample')
    ax[1].set_ylabel('Saliency')
    ax[1].set_title('Saliency after generator training')
    ax[0].set_yscale('symlog', linthresh=1e-3)
    ax[1].set_yscale('symlog', linthresh=1e-3)
    ax[0].set_ylim(-1e2, 1e2)
    ax[1].set_ylim(-1e2, 1e2)
    ax[0].grid()
    ax[1].grid()
    fig.suptitle('Key: %x  ;  Step: %d'%(key, step))
    plt.tight_layout()
    
    (fig, ax) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))
    disc_saliency = np.array(disc_saliency).flatten()
    gen_saliency = np.array(gen_saliency).flatten()
    bins = []
    bins.extend([-x for x in np.logspace(2, -3, 50)])
    bins.extend([0])
    bins.extend([x for x in np.logspace(-3, 2, 50)])
    ax[0].hist(disc_saliency, bins=bins, log=True, color='blue')
    ax[1].hist(gen_saliency, bins=bins, log=True, color='blue')
    ax[0].set_xscale('symlog', linthresh=1e-2)
    ax[1].set_xscale('symlog', linthresh=1e-2)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[0].set_xlim(-1e2, 1e2)
    ax[1].set_xlim(-1e2, 1e2)
    ax[0].set_xlabel('Saliency')
    ax[0].set_ylabel('Count')
    ax[1].set_xlabel('Saliency')
    ax[1].set_ylabel('Count')
    ax[0].set_title('Saliency after discriminator training')
    ax[1].set_title('Saliency after generator training')
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