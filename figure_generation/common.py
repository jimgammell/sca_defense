import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import get_package_module_names, get_package_modules
import figure_generation

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

def get_param_histograms(fig, axes, results):
    results_flat = flatten_results(results)
    gen_hist_bins = [r['gen_hist']['bin_edges'] for r in results_flat]
    gen_hists = [r['gen_hist']['histogram'] for r in results_flat]
    disc_hist_bins = [r['disc_hist']['bin_edges'] for r in results_flat]
    disc_hists = [r['disc_hist']['histogram'] for r in results_flat]
    xmin = np.min((np.min(disc_hist_bins), np.min(gen_hist_bins)))
    xmax = np.max((np.max(disc_hist_bins), np.max(gen_hist_bins)))
    xabs = np.max((xmin, xmax))
    xmin = -xabs
    xmax = xabs
    ymin = 0
    ymax = np.max((np.max(disc_hists), np.max(gen_hists)))
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_yscale('symlog', linthresh=1.)
    progress_bar = tqdm(desc='Plotting parameter histograms...', total=len(results_flat)+1, unit='frames')
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
    return lambda folder: anim.save(os.path.join(folder, 'param_hist.gif'), writer='ffmpeg', fps=10)

def plot_sampled_images(fig, axes, results):
    results_flat = flatten_results(results)
    try:
        sampled_images = np.array([r['sampled_gen_images']['protected_images'] for r in results_flat])
    except:
        sampled_images = np.array([r['sampled_gen_images']['fake_images'] for r in results_flat])
    sampled_images *= .5
    sampled_images += .5
    try:
        labels = [r['sampled_gen_images']['labels'] for r in results_flat]
    except:
        labels = None
    progress_bar = tqdm(desc='Plotting sampled images...', total=len(sampled_images)+1, unit='frames')
    plots = []
    def get_frame(t):
        nonlocal plots
        for plot in plots:
            plot.remove()
        plots = []
        for (ax, image, label) in zip(axes.flatten(), sampled_images[t], labels[t] if labels != None else len(axes.flatten())*[None]):
            plots.append(ax.imshow(np.transpose(image, (1, 2, 0)), cmap='binary', aspect='equal', interpolation='none'))
            if labels != None:
                ax.set_title('Label: {}'.format(label))
        fig.suptitle('Epoch: {}'.format(t))
        progress_bar.update(1)
        return axes
    anim = FuncAnimation(fig, get_frame, frames=len(sampled_images))
    return lambda folder: anim.save(os.path.join(folder, 'sampled_images.gif'), writer='ffmpeg', fps=10)

def plot_saliency(fig, axes, results):
    results_flat = flatten_results(results)
    sampled_saliencies = np.array([r['sampled_saliency']['saliency'] for r in results_flat])
    labels = np.array([r['sampled_saliency']['labels'] for r in results_flat])
    progress_bar = tqdm(desc='Plotting saliency...', total=len(sampled_saliencies)+1, unit='frames')
    plots = []
    def get_frame(t):
        nonlocal plots
        for plot in plots:
            plot.remove()
        plots = []
        for (ax, image, label) in zip(axes.flatten(), sampled_saliencies[t], labels[t]):
            plots.append(ax.imshow(np.transpose(image, (1, 2, 0)), cmap='plasma', aspect='equal', interpolation='none'))
            ax.set_title('Label: {}'.format(label))
        fig.suptitle('Epoch: {}'.format(t))
        progress_bar.update(1)
        return axes
    anim = FuncAnimation(fig, get_frame, frames=len(sampled_saliencies))
    return lambda folder: anim.save(os.path.join(folder, 'sampled_saliencies.gif'), writer='ffmpeg', fps=10)

def plot_confusion_matrices(fig, axes, results):
    results_flat = flatten_results(results)
    training_matrices = [r['training_metrics']['disc_conf_mtx'] for r in results_flat]
    testing_matrices = [r['test_metrics']['disc_conf_mtx'] for r in results_flat]
    progress_bar = tqdm(desc='Plotting confusion matrices...', total=len(results_flat)+1, unit='frames')
    plots = []
    def get_frame(t):
        nonlocal plots
        for plot in plots:
            plot.remove()
        plots = []
        plots.append(axes[0].imshow(training_matrices[t], cmap='binary', aspect='equal', interpolation='none'))
        plots.append(axes[1].imshow(testing_matrices[t], cmap='binary', aspect='equal', interpolation='none'))
        fig.suptitle('Epoch: {}'.format(t))
        progress_bar.update(1)
        return axes
    anim = FuncAnimation(fig, get_frame, frames=len(results_flat))
    return lambda folder: anim.save(os.path.join(folder, 'confusion_matrices.gif'), writer='ffmpeg', fps=10)

def main(trial_dir):
    with open(os.path.join(trial_dir, 'config.json'), 'r') as F:
        settings = json.load(F)
    with open(os.path.join(trial_dir, 'training_metrics.pickle'), 'rb') as F:
        results = pickle.load(F)
    if not os.path.exists(os.path.join(trial_dir, 'figures')):
        os.mkdir(os.path.join(trial_dir, 'figures'))
    trial_type = settings['trial']
    gen_figs_md = get_package_modules(figure_generation)[
        get_package_module_names(figure_generation)[0].index(trial_type)]
    gen_figs_fn = gen_figs_md.main
    figure_save_fns = gen_figs_fn(results, settings)
    for figure_save_fn in figure_save_fns:
        figure_save_fn(os.path.join(trial_dir, 'figures'))
    plt.close('all')