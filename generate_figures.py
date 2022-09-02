import os
import json
import pickle
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def flatten_results(results):
    return sum([[sub_trial_item for _, sub_trial_item in sub_trial_results.items()]
                for _, sub_trial_results in results.items()], [])

def get_epochs(results):
    n_epochs = sum(n for _, _, n in results.keys())
    epochs = np.arange(0, n_epochs)
    return epochs

def get_trial_boundaries(results):
    trial_types = [t for _, t, _ in results.keys()]
    sub_trial_epochs = np.array([0] + [n for _, _, n in results.keys()]).astype(int)
    sub_trial_epochs[1] -= 1
    boundary_epochs = np.cumsum(sub_trial_epochs)
    return trial_types, boundary_epochs

def color_trial_types(ax, results, alpha=.1):
    trial_types, boundary_epochs = get_trial_boundaries(results)
    for trial_type, e0, e1 in zip(trial_types, boundary_epochs[:-1], boundary_epochs[1:]):
        if trial_type == 'd':
            color = 'red'
        elif trial_type == 'g':
            color = 'blue'
        elif ('d' in trial_type and 'g' in trial_type):
            color = 'purple'
        else:
            assert False
        ax.axvspan(e0, e1, alpha=alpha, color=color)

def get_training_traces(results):
    results_flat = flatten_results(results)
    gen_loss = [r['training_metrics']['gen_loss'] for r in results_flat]
    disc_loss_fake_train = [r['training_metrics']['disc_loss_fake'] for r in results_flat]
    disc_loss_real_train = [r['training_metrics']['disc_loss_real'] for r in results_flat]
    disc_acc_fake_train = [r['training_metrics']['disc_acc_fake'] for r in results_flat]
    disc_acc_real_train = [r['training_metrics']['disc_acc_real'] for r in results_flat]
    disc_loss_real_test = [r['disc_generalization']['disc_loss'] for r in results_flat]
    disc_acc_real_test = [r['disc_generalization']['disc_acc'] for r in results_flat]
    return {'Generator loss': gen_loss,
            'Discriminator loss (fake/train)': disc_loss_fake_train,
            'Discriminator loss (real/train)': disc_loss_real_train,
            'Discriminator loss (real/test)': disc_loss_real_test,
            'Discriminator accuracy (fake/train)': disc_acc_fake_train,
            'Discriminator accuracy (real/train)': disc_acc_real_train,
            'Discriminator accuracy (real/test)': disc_acc_real_test}

def plot_training_traces(fig, axes, results):
    def get_color(label):
        if 'Generator' in label:
            return 'blue'
        elif 'Discriminator' in label:
            return 'red'
        else:
            assert False
    def get_linestyle(label):
        if 'fake/train' in label:
            return 'dotted'
        elif 'real/train' in label:
            return 'dashed'
        elif 'real/test' in label:
            return 'solid'
        else:
            return 'solid'
    def get_axis(label):
        if 'Generator' in label:
            return axes[0]
        elif 'Discriminator loss' in label:
            return axes[1]
        elif 'Discriminator accuracy' in label:
            return axes[2]
        else:
            assert False
    epochs = get_epochs(results)
    traces = get_training_traces(results)
    for trace_label, trace in traces.items():
        ax = get_axis(trace_label)
        color = get_color(trace_label)
        linestyle = get_linestyle(trace_label)
        ax.plot(epochs, trace, color=color, linestyle=linestyle, label=trace_label)
    return lambda dest: fig.savefig(dest)

def generate_param_histograms(fig, axes, results):
    results_flat = flatten_results(results)
    gen_hist_bins = [r['gen_hist']['bin_edges'] for r in results_flat]
    gen_hists = [r['gen_hist']['histogram'] for r in results_flat]
    disc_hist_bins = [r['disc_hist']['bin_edges'] for r in results_flat]
    disc_hists = [r['disc_hist']['histogram'] for r in results_flat]
    xmin = np.min((np.min(disc_hist_bins), np.min(gen_hist_bins)))
    xmax = np.max((np.max(disc_hist_bins), np.max(gen_hist_bins)))
    xabs = np.max((np.abs(xmin), np.abs(xmax)))
    xmin = -xabs
    xmax = xabs
    ymin = 0
    ymax = np.max((np.max(disc_hists), np.max(gen_hists)))
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_yscale('symlog', linthresh=1.)
    
    progress_bar = tqdm(total=len(results_flat)+1)
    hists = []
    def get_frame(t):
        nonlocal hists
        for hist in hists:
            hist.remove()
        hists = []
        hists.append(axes[0].bar(gen_hist_bins[t][:-1], gen_hists[t], width=np.diff(gen_hist_bins[t]), color='blue'))
        hists.append(axes[1].bar(disc_hist_bins[t][:-1], disc_hists[t], width=np.diff(disc_hist_bins[t]), color='red'))
        fig.suptitle('Epoch: {}'.format(t))
        progress_bar.update(1)
        return axes
    anim = FuncAnimation(fig, get_frame, frames=len(results_flat))
    return lambda dest: anim.save(dest, writer='ffmpeg', fps=10)

def plot_sampled_images(fig, axes, results):
    results_flat = flatten_results(results)
    sampled_images = [r['sampled_gen_images']['fake_images'] for r in results_flat]
    progress_bar = tqdm(total=len(sampled_images)+1)
    plots = []
    def get_frame(t):
        nonlocal plots
        for plot in plots:
            plot.remove()
        plots = []
        for (ax, image) in zip(axes.flatten(), sampled_images[t]):
            plots.append(ax.imshow(np.transpose(image, (1, 2, 0)), cmap='binary', aspect='equal', interpolation='none'))
        fig.suptitle('Epoch: {}'.format(t))
        progress_bar.update(1)
        return axes
    anim = FuncAnimation(fig, get_frame, frames=len(sampled_images))
    return lambda dest: anim.save(dest, writer='ffmpeg', fps=10)
        
def generate_gan_figures(results):
    
    training_curves_fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 24))
    training_curve_save_fn = plot_training_traces(training_curves_fig, axes, results)
    for ax in axes:
        color_trial_types(ax, results)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid()
        epochs = get_epochs(results)
        ax.set_xlim(0, len(epochs)-1)
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('Loss')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(-.05, 1.05)
    axes[0].set_title('Generator')
    axes[1].set_title('Discriminator')
    axes[2].set_title('Discriminator')
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    plt.tight_layout()
    
    hist_fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_xlabel('Parameter value')
    axes[1].set_xlabel('Parameter value')
    axes[0].set_ylabel('Count')
    axes[1].set_ylabel('Count')
    axes[0].set_title('Generator')
    axes[1].set_title('Discriminator')
    hist_anim_save_fn = generate_param_histograms(hist_fig, axes, results)
    
    sampled_images_fig, axes = plt.subplots(4, 8, figsize=(2*8, 2*4), sharex=True, sharey=True)
    for ax in axes.flatten():
        ax.axis('off')
    sampled_images_fig.suptitle('Epoch: 0')
    plt.tight_layout()
    sampled_images_save_fn = plot_sampled_images(sampled_images_fig, axes, results)
    
    return training_curve_save_fn, hist_anim_save_fn, sampled_images_save_fn

def main(trial_dir):
    with open(os.path.join(trial_dir, 'config.json'), 'r') as F:
        settings = json.load(F)
    with open(os.path.join(trial_dir, 'training_metrics.pickle'), 'rb') as F:
        results = pickle.load(F)
    
    if not os.path.exists(os.path.join(trial_dir, 'figures')):
        os.mkdir(os.path.join(trial_dir, 'figures'))
    
    trial_type = settings['trial']
    if trial_type == 'train_gan':
        training_curve_save_fn, hist_anim_save_fn, sampled_images_save_fn = generate_gan_figures(results)
    
    training_curve_save_fn(os.path.join(trial_dir, 'figures', 'training_results.png'))
    hist_anim_save_fn(os.path.join(trial_dir, 'figures', 'param_hist.gif'))
    sampled_images_save_fn(os.path.join(trial_dir, 'figures', 'sampled_images.gif'))