import os
from matplotlib import pyplot as plt
import numpy as np

from figure_generation.common import flatten_results, get_epochs, get_trial_boundaries, get_param_histograms, plot_sampled_images, plot_confusion_matrices, plot_saliency

def plot_training_traces(fig, axes, results, settings):
    flat_results = flatten_results(results)
    gen_loss_train = [r['training_metrics']['gen_loss'] for r in flat_results]
    gen_loss_test = [r['test_metrics']['gen_loss'] for r in flat_results]
    disc_loss_train = [r['training_metrics']['disc_loss'] for r in flat_results]
    disc_loss_test = [r['test_metrics']['disc_loss'] for r in flat_results]
    disc_loss_train_ind = np.array([r['ind_disc_metrics']['train_loss'][-1] for r in flat_results if 'ind_disc_metrics' in r.keys()])
    disc_loss_test_ind = np.array([r['ind_disc_metrics']['test_loss'][-1] for r in flat_results if 'ind_disc_metrics' in r.keys()])
    disc_acc_train = [r['training_metrics']['disc_acc'] for r in flat_results]
    disc_acc_test = [r['test_metrics']['disc_acc'] for r in flat_results]
    disc_acc_train_ind = np.array([r['ind_disc_metrics']['train_acc'][-1] for r in flat_results if 'ind_disc_metrics' in r.keys()])
    disc_acc_test_ind = np.array([r['ind_disc_metrics']['test_acc'][-1] for r in flat_results if 'ind_disc_metrics' in r.keys()])
    epochs = get_epochs(results)
    ind_epochs = np.linspace(0, np.max(epochs), len(disc_loss_train_ind))
    
    axes[0].plot(epochs, gen_loss_train, color='blue', linestyle='--', label='Train')
    axes[0].plot(epochs, gen_loss_test, color='blue', linestyle='-', label='Test')
    axes[1].plot(epochs, disc_loss_train, color='red', linestyle='--', label='Train')
    axes[1].plot(epochs, disc_loss_test, color='red', linestyle='-', label='Test')
    axes[1].plot(ind_epochs, disc_loss_train_ind,
                 color='black', linestyle='none', marker='o', label='Train-Independent', markersize=5)
    axes[1].plot(ind_epochs, disc_loss_test_ind,
                 color='black', linestyle='none', marker='x', label='Test-Independent', markersize=5)
    axes[2].plot(epochs, disc_acc_train, color='red', linestyle='--', label='Train')
    axes[2].plot(epochs, disc_acc_test, color='red', linestyle='-', label='Test')
    axes[2].plot(ind_epochs, disc_acc_train_ind,
                 color='black', linestyle='none', marker='o', label='Train-Independent', markersize=5)
    axes[2].plot(ind_epochs, disc_acc_test_ind,
                 color='black', linestyle='none', marker='x', label='Test-Independent', markersize=5)

def main(results, settings):
    n_saliencies = 32#len(flatten_results(results)[0]['sampled_saliency']['saliency'])
    fig, axes = plt.subplots(4, n_saliencies//4, figsize=(2*n_saliencies//4, 2*4))
    for ax in axes.flatten():
        ax.axis('off')
    fig.suptitle('Epoch: 0')
    plt.tight_layout()
    saliency_anim_save_fn = plot_saliency(fig, axes, results)
    
    traces_fig, axes = plt.subplots(3, 1, figsize=(8, 24))
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('Loss')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(-.05, 1.05)
    axes[0].set_title('Generator')
    axes[1].set_title('Discriminator')
    axes[2].set_title('Discriminator')
    plot_training_traces(traces_fig, axes, results, settings)
    for ax in axes:
        ax.set_xlabel('Epoch')
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
    hist_anim_save_fn = get_param_histograms(fig, axes, results)
    
    n_images = 32#len(flatten_results(results)[0]['sampled_gen_images']['protected_images'])
    fig, axes = plt.subplots(4, n_images//4, figsize=(2*n_images//4, 2*4))
    for ax in axes.flatten():
        ax.axis('off')
    fig.suptitle('Epoch: 0')
    plt.tight_layout()
    gen_img_anim_save_fn = plot_sampled_images(fig, axes, results)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_xlabel('Predicted value')
    axes[0].set_ylabel('True value')
    axes[1].set_xlabel('Predicted value')
    axes[1].set_ylabel('True value')
    axes[0].set_title('Training dataset')
    axes[1].set_title('Holdout dataset')
    conf_mtx_anim_save_fn = plot_confusion_matrices(fig, axes, results)
    
    return (trace_plot_save_fn, hist_anim_save_fn, gen_img_anim_save_fn, saliency_anim_save_fn, conf_mtx_anim_save_fn)