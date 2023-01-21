import os
import pickle
import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from ray import tune
import torch
from tqdm import tqdm
from copy import deepcopy

def get_figsize(rows, cols):
    W = H = 4
    return (W*cols, H*rows)

def default_color():
    return 'blue'

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

def load_traces(base_dir, epochs=None, phases=None):
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
        test_epochs = [int(s.split('.')[0].split('_')[-1]) for s in test_results_files]
    train_indices = np.argsort(epochs)
    test_indices = np.argsort(test_epochs)
    train_keys = [k for k in load_results_file(train_results_files[0]).keys()]
    test_keys = [k for k in load_results_file(test_results_files[0]).keys()]
    traces = {
        'epochs': np.array(epochs)[train_indices],
        **{'train_'+str(key): np.array([load_results_file(f)[key] for f in train_results_files])[train_indices] for key in train_keys},
        **{'test_'+str(key): np.array([load_results_file(f)[key] for f in test_results_files])[test_indices] for key in test_keys}
    }
    return traces

def get_metrics_from_results_grid(results_grid):
    metrics = {}
    for result in results_grid:
        for metric_name, metric_val in result.metrics.items():
            if type(metric_val) not in [float, np.float32]:
                continue
            if metric_name not in metrics.keys():
                metrics[metric_name] = []
            metrics[metric_name].append(result.metrics[metric_name])
    return metrics

def hsweep_metrics_hist(results_grid, **kwargs):
    metrics = get_metrics_from_results_grid(results_grid)
    fig, axes = plt.subplots(1, len(metrics), figsize=get_figsize(1, len(metrics)))
    for (metric_name, metric_vals), ax in zip(metrics.items(), axes):
        ax.hist(metric_vals, bins=len(metric_vals), cumulative=True, **kwargs)
        ax.set_title(metric_name)
        ax.set_xlabel('Best value attained')
        ax.set_ylabel('Cumulative # trials')
    fig.suptitle('Histograms of best values')
    plt.tight_layout()
    return fig, axes

def hsweep_all_training_curves(results_grid, **kwargs):
    metrics = get_metrics_from_results_grid(results_grid)
    fig, axes = plt.subplots(1, len(metrics), figsize=get_figsize(1, len(metrics)))
    for ax, metric_name in zip(axes, metrics.keys()):
        for idx, result in enumerate(results_grid):
            result.metrics_dataframe.plot(
                'training_iteration', metric_name,             
                ax=ax, color=default_color(), legend=False, **kwargs)
        best_result = results_grid.get_best_result('mean_rank', 'min')
        best_result.metrics_dataframe.plot(
            'training_iteration', metric_name,
            ax=ax, color='red', **kwargs)
        ax.legend(handles=[
            mpatches.Patch(color='red', label='Best trial'),
            mpatches.Patch(color=default_color(), label='Some trial')])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title(metric_name)
    fig.suptitle('All training curves')
    plt.tight_layout()
    return fig, axes

def hsweep_metric_vs_hparam(results_grid, **kwargs):
    metrics = get_metrics_from_results_grid(results_grid)
    hparams = {}
    for result in results_grid:
        for config_name, config_val in result.config.items():
            if config_name not in hparams.keys():
                hparams[config_name] = {}
            if 'hparam_val' not in hparams[config_name].keys():
                hparams[config_name]['hparam_val'] = []
            hparams[config_name]['hparam_val'].append(config_val)
            for metric_name, metric_val in result.metrics.items():
                if type(metric_val) not in [float, np.float32]:
                    continue
                if metric_name not in hparams[config_name].keys():
                    hparams[config_name][metric_name] = []
                hparams[config_name][metric_name].append(metric_val)
    fig, axes = plt.subplots(len(hparams), len(metrics), figsize=get_figsize(len(hparams), len(metrics)))
    for hparam_idx, hparam_name in enumerate(hparams.keys()):
        axes[hparam_idx][0].set_ylabel(hparam_name)
        for metric_idx, metric_name in enumerate([k for k in hparams[hparam_name].keys() if k != 'hparam_val']):
            ax = axes[hparam_idx][metric_idx]
            ax.plot(hparams[hparam_name]['hparam_val'], hparams[hparam_name][metric_name], '.', **kwargs)
            axes[0][metric_idx].set_title(metric_name)
    fig.suptitle('Performance vs. hyperparameter value')
    plt.tight_layout()
    return fig, axes

def visualize_gaussian_dataset(num_samples=4):
    from datasets.toy_datasets import GaussianDataset
    (fig, axes) = plt.subplots(1, num_samples, figsize=get_figsize(1, num_samples), sharex=True, sharey=True)
    for idx, ax in enumerate(axes):
        dataset = GaussianDataset(use_existing_distribution=False)
        class0_samples = dataset.x[dataset.y==0]
        class1_samples = dataset.x[dataset.y==1]
        class0_useful_features = class0_samples[:, :2]
        class1_useful_features = class1_samples[:, :2]
        ax.plot(class0_useful_features[:, 0], class0_useful_features[:, 1],
                '.', color='blue', label='Class 0')
        ax.plot(class1_useful_features[:, 0], class1_useful_features[:, 1],
                '.', color='red', label='Class 1')
        ax.legend()
        ax.set_xlabel('Useful feature 1')
        ax.set_ylabel('Useful feature 2')
        ax.set_title('Sample: %d'%(idx))
    fig.suptitle('Projection of samples onto dimensions of useful features')
    figs_to_save = {'dataset_visualization': fig}
    return figs_to_save

def plot_gaussian_decision_boundary(results_dir):
    from datasets.toy_datasets import GaussianDataset
    from models.toy_models import LinearModel
    train_dataset = GaussianDataset()
    test_dataset = GaussianDataset()
    checkpoint_files = [f for f in os.listdir(results_dir) if 'checkpoint' in f]
    checkpoint_timesteps = [int(f.split('_')[-1]) for f in checkpoint_files]
    final_model_dir = 'checkpoint_'+str(np.max(checkpoint_timesteps))
    final_model_state = torch.load(os.path.join(results_dir, final_model_dir, 'disc_state.pth'))
    final_model = LinearModel(2, 2)
    final_model.load_state_dict(final_model_state)
    (fig, axes) = plt.subplots(1, 2, figsize=get_figsize(1, 2), sharex=True, sharey=True)
    for dataset, ax in zip([train_dataset, test_dataset], axes):
        class0_samples = dataset.x[dataset.y==0]
        class1_samples = dataset.x[dataset.y==1]
        class0_useful_features = class0_samples[:, :2]
        class1_useful_features = class1_samples[:, :2]
        ax.plot(class0_useful_features[:, 0], class0_useful_features[:, 1],
                '.', color='blue', label='Class 0')
        ax.plot(class1_useful_features[:, 0], class1_useful_features[:, 1],
                '.', color='red', label='Class 1')
        ax.legend()
        ax.set_xlabel('Useful feature 1')
        ax.set_ylabel('Useful feature 2')
        
    xlims = axes[0].get_xlim()
    ylims = axes[0].get_ylim()
    samples_per_axis = 1000
    pred = np.zeros((samples_per_axis, samples_per_axis))
    xx = np.linspace(xlims[0], xlims[1], pred.shape[0])
    yy = np.linspace(ylims[0], ylims[1], pred.shape[1])
    pbar = tqdm(total=np.prod(pred.shape))
    for ridx, x in enumerate(xx):
        for cidx, y in enumerate(yy):
            logits = final_model(torch.tensor([x, y]).to(torch.float))
            pred[ridx, cidx] = 1.0 if logits[0]>=logits[1] else -1.0
            pbar.update(1)
    xx, yy = np.meshgrid(xx, yy)
    axes[0].contourf(xx, yy, pred, colors=['red', 'blue'], alpha=0.1)
    axes[1].contourf(xx, yy, pred, colors=['red', 'blue'], alpha=0.1)
    
    axes[0].set_title('Training dataset')
    axes[1].set_title('Testing dataset')
    fig.suptitle('Model decision boundary')
    return {'decision_boundary': fig}

def animate_decision_boundary(results_dir):
    frames_path = os.path.join(results_dir, 'decision_boundary_frames')
    if not os.path.exists(frames_path):
        return
    from PIL import Image
    image_files = [f for f in os.listdir(frames_path) if '.png' in f]
    sorted_indices = np.argsort([int(f.split('.')[0].split('_')[1]) for f in image_files])
    image_files = [image_files[idx] for idx in sorted_indices]
    images = [Image.open(os.path.join(frames_path, f)) for f in image_files]
    images[0].save(os.path.join(results_dir, 'decision_boundary_over_time.gif'),
                   format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)

def basic_eval(results_dir):
    figs_to_save = {}       
    traces = load_traces(results_dir)
    epochs = traces['epochs']
    metrics = set([('_'.join(k.split('_')[1:]) if 'train' in k or 'test' in k else k)
                   for k in traces.keys() if k != 'epoch'])
    scalar_metrics = [m for m in ['loss', 'accuracy', 'mean_rank', 'disc_loss', 'disc_accuracy', 'gen_loss', 'gen_oracle_loss'] if m in metrics]
    sc_fig, sc_axes = plt.subplots(
        1, len(scalar_metrics), figsize=get_figsize(1, len(scalar_metrics)), sharex=True)
    if not hasattr(sc_axes, '__iter__'):
        sc_axes = [sc_axes]
    for ax, scalar_metric in zip(sc_axes, scalar_metrics):
        if 'train_'+scalar_metric in traces.keys():
            plot_scalar_trace(epochs, traces['train_'+scalar_metric], fig=sc_fig, ax=ax,
                              color=default_color(), label=scalar_metric+' (train)', linestyle='--')
        if 'test_'+scalar_metric in traces.keys():
            plot_scalar_trace(epochs, traces['test_'+scalar_metric], fig=sc_fig, ax=ax,
                              color=default_color(), label=scalar_metric+' (test)', linestyle='-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title(scalar_metric)
        ax.legend()
    plt.tight_layout()
    figs_to_save['scalar_metrics'] = sc_fig
    return figs_to_save

def basic_hsweep(results_dir):
    figs_to_save = {}
    ray_tuner = tune.Tuner.restore(results_dir)
    results_grid = ray_tuner.get_results()
    fig, _ = hsweep_metrics_hist(results_grid, color=default_color())
    figs_to_save['hsweep_results_hist'] = fig
    fig, _ = hsweep_all_training_curves(results_grid)
    figs_to_save['hsweep_training_curves'] = fig
    fig, _ = hsweep_metric_vs_hparam(results_grid, color=default_color())
    figs_to_save['hsweep_metrics_vs_hparam'] = fig
    return figs_to_save

def generate_figures(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    if 'eval' in results_dir:
        figs_to_save = basic_eval(results_dir)
    elif 'toy_gaussian_classification' in results_dir:
        figs_to_save = basic_eval(results_dir)
        figs_to_save.update(plot_gaussian_decision_boundary(results_dir))
    elif 'toy_gaussian_adversarial' in results_dir:
        figs_to_save = basic_eval(results_dir)
        animate_decision_boundary(results_dir)
    elif 'hsweep' in results_dir:
        figs_to_save = basic_hsweep(results_dir)
    elif 'visualize' in results_dir:
        if 'toy_gaussian' in results_dir:
            figs_to_save = visualize_gaussian_dataset()
        else:
            assert False
    else:
        assert False
    for fig_name, fig in figs_to_save.items():
        fig.savefig(os.path.join(results_dir, fig_name+'.pdf'))
        fig.savefig(os.path.join(results_dir, fig_name+'.jpg'))
    