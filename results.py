import os
import pickle
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

def get_trace(results_dict):
    loss_dict = gen_training_loss[key]
    loss_dict.update({gen_training_loss[-1]: None})
    epochs = []
    losses = []
    for (e1, e2) in zip(loss_dict[:-1], loss_dict[1:]):
        epochs.extend([x for x in np.linspace(e1, e2, len(loss_dict[e1]))])
        losses.extend(loss_dict[e1])
    epochs = np.array(epochs)
    losses = np.array(losses)
    return (epochs, losses)

def display_results(results):
    out = {
        'loss_over_time': None,
        'saliency': []}
    
    # Plot loss over time
    (fig, ax) = plt.subplots(2, 1, sharex=True, figsize=(16, 16))
    gen_training_loss = results['gen_training_loss']
    colors = []
    while len(colors) < len(gen_training_loss):
        colors.extend(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for (key, color) in zip(gen_training_loss, colors):
        results_dict = gen_training_loss[key]
        (epochs, losses) = get_trace(results_dict)
        ax[0].plot(epochs, losses, color=color, label='%x'%(key))
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Generator training')
    ax[0].legend()
    disc_training_loss = results['disc_training_loss']
    (epochs, losses) = get_trace(disc_training_loss[disc_training_loss.keys()[0]])
    for key in disc_training_loss.keys()[1:]:
        (_, loss) = get_trace(disc_training_loss[key])
        losses += loss
    losses /= len(disc_training_loss)
    ax[1].plot(epochs, losses, color='blue')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Discriminator training')
    fig.suptitle('GAN loss over time')
    out['loss_over_time'] = fig
    
    # Plot saliency curves
    cmap = cm('seismic')
    for (idx, key) in enumerate(results['saliency']):
        for (original_trace, saliency) in results['saliency'][key]:
            saliency /= np.max((np.abs(np.max(saliency)), np.abs(np.min(saliency))))
            def get_color(point):
                return cmap(1-point)
            (fig, ax) = plt.subplots(2, 1, sharex=True, figsize=(16, 16))
            ax[0].plot(range(len(original_trace)), original_trace, '.', markersize=.5, color='blue')
            for ((x, i), y) in zip(enumerate(saliency), original_trace):
                ax[1].plot((x), (y), '.', markersize=.5, color=get_color(i))
            ax[0].set_xlabel('Sample')
            ax[1].set_xlabel('Sample')
            ax[0].set_ylabel('Amplitude')
            ax[1].set_ylabel('Amplitude')
            ax[0].set_title('Original trace')
            ax[1].set_title('Saliency')
            fig.suptitle('Key: %x , Example: %d'%(key, idx))
            out['saliency'].append(fig)
    
    return out

def save_results(figures, results, dest):
    for fig_key in figures:
        fig = figures[fig_key]
        fig.savefig(os.path.join(os.getcwd(), dest, 'fig_%s.png'%(fig_key)))
    with open(os.path.join(os.getcwd(), dest, 'results.pickle'), 'wb') as F:
        pickle.dump(results, F)