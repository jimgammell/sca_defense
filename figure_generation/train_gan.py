import os
from matplotlib import pyplot as plt
import numpy as np

from figure_generation.common import flatten_results, get_epochs, get_trial_boundaries, get_param_histograms, plot_sampled_images, plot_confusion_matrices

def plot_training_traces(fig, axes, results, settings):
    flat_results = flatten_results(results)
    gen_loss = np.array([r['training_metrics']['gen_loss'] for r in flat_results])
    disc_loss_fake_train = np.array([r['training_metrics']['disc_loss_fake'] for r in flat_results])
    disc_loss_real_train = np.array([r['training_metrics']['disc_loss_real'] for r in flat_results])
    disc_acc_fake_train = np.array([r['training_metrics']['disc_acc_fake'] for r in flat_results])
    disc_acc_real_train = np.array([r['training_metrics']['disc_acc_real'] for r in flat_results])
    disc_loss_real_test = np.array([r['disc_generalization']['disc_loss'] for r in flat_results])
    disc_acc_real_test = np.array([r['disc_generalization']['disc_acc'] for r in flat_results])
    
    epochs = get_epochs(results)
    axes[0].plot(epochs, gen_loss, color='blue', label='Fake/test')
    axes[1].plot(epochs, disc_loss_fake_train, color='red', linestyle='dotted', label='Fake/train')
    axes[1].plot(epochs, disc_loss_real_train, color='red', linestyle='dashed', label='Real/train')
    axes[1].plot(epochs, disc_loss_real_test, color='red', linestyle='solid', label='Real/test')
    axes[2].plot(epochs, disc_acc_fake_train, color='red', linestyle='dotted', label='Fake/train')
    axes[2].plot(epochs, disc_acc_real_train, color='red', linestyle='dashed', label='Real/train')
    axes[2].plot(epochs, disc_acc_real_test, color='red', linestyle='solid', label='Real/test')

def main(results, settings):
    traces_fig, axes = plt.subplots(3, 1, figsize=(8, 24))
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('Loss')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0, 1)
    axes[0].set_title('Generator')
    axes[1].set_title('Discriminator')
    axes[2].set_title('Discriminator')
    plot_training_traces(traces_fig, axes, results, settings)
    for ax in axes:
        ax.set_label('Epoch')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    trace_plot_save_fn = lambda folder: traces_fig.savefig(os.path.join(folder, 'training_curves.png'))
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_xlabel('Parameter value')
    axes[1].set_xlabel('Parameter value')
    axes[0].set_ylabel('Count')
    axes[1].set_ylabel('Count')
    axes[0].set_title('Generator')
    axes[1].set_title('Discriminator')
    fig.suptitle('Epoch: 0')
    plt.tight_layout()
    hist_anim_save_fn = get_param_histograms(fig, axes, results)
    
    batch_size = settings['dataloader_kwargs']['batch_size']
    fig, axes = plt.subplots(batch_size//8, 8, figsize=(2*8, 2*batch_size//8))
    for ax in axes.flatten():
        ax.axis('off')
    fig.suptitle('Epoch: 0')
    plt.tight_layout()
    gen_img_anim_save_fn = plot_sampled_images(fig, axes, results)
    
    return (trace_plot_save_fn, hist_anim_save_fn, gen_img_anim_save_fn)