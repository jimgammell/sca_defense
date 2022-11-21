import os
import pickle
from copy import deepcopy
import numpy as np
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM as GC

def get_trace(results_path, files, criteria, key):
    epochs = []
    trace = []
    for file in files:
        if not criteria(file):
            continue
        epoch = int(file.split('.')[-2].split('_')[-1])
        epochs.append(epoch)
        with open(os.path.join(results_path, file), 'rb') as F:
            results = pickle.load(F)
        trace.append(results[key])
    indices = np.argsort(epochs)
    trace = np.array(trace)[indices]
    trace = np.concatenate(trace)
    return trace

def show_dataset(dataset, indices=None, examples_per_class=10, classes=None, plot_width=4, plot_height=4):
    if classes is None:
        unique_labels = np.unique([int(x) for x in dataset.labels])
        np.random.shuffle(unique_labels)
        classes = unique_labels[:examples_per_class]
    if indices is None:
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
    remaining_to_show = {c: examples_per_class for c in classes}
    fig, axes = plt.subplots(examples_per_class, len(classes), figsize=(plot_width*len(classes), plot_height*examples_per_class),
                             sharex=True, sharey=True)
    displayed_indices = []
    for idx in indices:
        trace, label = dataset[idx]
        #if overlay_fn is not None:
        #    overlay = overlay_fn(trace)
        #if transform is not None:
        #    image = transform(image)
        trace = trace.squeeze()
        label = int(label)
        if label in classes and remaining_to_show[label] > 0:
            remaining_to_show[label] -= 1
            ax = axes[remaining_to_show[label], np.where(classes==label)][0][0]
            ax.plot(np.arange(len(trace)), trace, color='blue')
            displayed_indices.append(idx)
    for ax in axes[:, 0]:
        ax.set_ylabel('Value')
    for ax in axes[-1, :]:
        ax.set_xlabel('Sample')
    for ax, c in zip(axes[0, :], classes):
        ax.set_title('Class: 0x%s'%(hex(c)))
    for ax in axes.flatten():
        ax.grid(True)
    return (fig, axes), displayed_indices

def plot_gradcam(dataset, model, target_layers, indices, use_cuda=False):
    grad_cam = GC(model, target_layers=target_layers, use_cuda=use_cuda)
    def get_gc_overlay(image):
        overlay = grad_cam(input_tensor=image)[0].reshape(image.squeeze().shape)
        return overlay
    for idx in indices:
        trace, label = dataset[idx]
        print(trace.shape)
        label = int(label)
        overlay = get_gc_overlay(trace)
        print(overlay.shape, np.min(overlay), np.max(overlay))
    
def plot_traces(results_path, file_prefix, keys, plot_size=4):
    files = [f for f in os.listdir(results_path) if f.split('__')[0] == file_prefix]
    assert len(files) != 0
    fig, axes = plt.subplots(1, len(keys), figsize=(len(keys)*plot_size, plot_size))
    if len(keys) == 1:
        axes = np.array([axes])
    for key, ax in zip(keys, axes):
        try:
            train_trace = get_trace(results_path, files,
                                    lambda f: 'train' in f.split('__')[1], key)
            train_epochs = np.linspace(0, 1, len(train_trace))
            ax.plot(train_epochs, train_trace, '.', color='blue', label='Train')
        except:
            pass
        eval_trace = get_trace(results_path, files,
                               lambda f: 'eval' in f.split('__')[1], key)
        eval_epochs = np.linspace(0, 1, len(eval_trace))
        ax.plot(eval_epochs, eval_trace, '.', color='red', label='Test')
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)
    return fig, axes

def _plot_traces(results_path, plot_width=4, plot_height=4):
    pretrain_gen_files = [f for f in os.listdir(results_path) if 'gen_pretrain' in f.split('__')[0]]
    pretrain_disc_files = [f for f in os.listdir(results_path) if 'disc_pretrain' in f.split('__')[0]]
    if len(pretrain_gen_files) != 0:
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        train_loss = get_trace(results_path, pretrain_gen_files,
                               lambda f: 'train' in f.split('__')[1], 'loss')
        eval_loss = get_trace(results_path, pretrain_gen_files,
                              lambda f: 'eval' in f.split('__')[1], 'loss')
        train_epochs = np.linspace(0, 1, len(train_loss))
        eval_epochs = np.linspace(0, 1, len(eval_loss))
        ax.plot(train_epochs, train_loss, '.', color='blue', label='Train')
        ax.plot(eval_epochs, eval_loss, '.', color='red', label='Test')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE loss')
        ax.legend()
        ax.grid(True)
        fig.suptitle('Generator pretraining curves')
    if len(pretrain_disc_files) != 0:
        fig, axes = plt.subplots(1, 3, figsize=(3*plot_height, plot_width))
        train_loss = get_trace(results_path, pretrain_disc_files,
                               lambda f: 'train' in f.split('__')[1], 'loss')
        eval_loss = get_trace(results_path, pretrain_disc_files,
                              lambda f: 'eval' in f.split('__')[1], 'loss')
        train_acc = get_trace(results_path, pretrain_disc_files,
                              lambda f: 'train' in f.split('__')[1], 'acc')
        eval_acc = get_trace(results_path, pretrain_disc_files,
                             lambda f: 'eval' in f.split('__')[1], 'acc')
        train_rank = get_trace(results_path, pretrain_disc_files,
                               lambda f: 'train' in f.split('__')[1], 'mean_rank')
        eval_rank = get_trace(results_path, pretrain_disc_files,
                              lambda f: 'eval' in f.split('__')[1], 'mean_rank')
        train_epochs = np.linspace(0, 1, len(train_loss))
        eval_epochs = np.linspace(0, 1, len(eval_loss))
        axes[0].plot(train_epochs, train_loss, '.', color='blue', label='Train')
        axes[0].plot(eval_epochs, eval_loss, '.', color='red', label='Test')
        axes[1].plot(train_epochs, 100*train_acc, '.', color='blue', label='Train')
        axes[1].plot(eval_epochs, 100*eval_acc, '.', color='red', label='Test')
        axes[2].plot(train_epochs, train_rank, '.', color='blue', label='Train')
        axes[2].plot(eval_epochs, eval_rank, '.', color='red', label='Test')
        for ax in axes.flatten():
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True)
        axes[0].set_ylabel('Cross entropy loss')
        axes[1].set_ylabel('Accuracy')
        axes[2].set_ylabel('Mean rank')
        fig.suptitle('Discriminator pretraining curves')