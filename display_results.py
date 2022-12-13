import numpy as np
import torch
from matplotlib import pyplot as plt

@torch.no_grad()
def plot_autoencoder_traces(traces, model, device, fig=None, axes=None):
    traces = traces.to(device)
    autoencoded_traces = model(traces).cpu().numpy()
    traces = traces.cpu().numpy()
    if fig is None:
        assert axes is None
        fig, axes = plt.subplots(traces.shape[0], 2, figsize=(8, 4*traces.shape[0]), sharex=True, sharey=True)
        if traces.shape[0] == 1:
            axes = np.expand_dims(axes, 0)
    for idx in range(len(traces)):
        axes[idx, 0].plot(traces[idx], color='blue', label='True trace')
        axes[idx, 0].plot(autoencoded_traces[idx], linestyle='--', color='red', label='Reconstructed trace')
        axes[idx, 1].plot(traces[idx]-autoencoded_traces[idx], color='blue')
    for ax in axes.flatten():
        ax.set_xlabel('Sample number')
    for ax in axes[:, 0]:
        ax.set_ylabel('Trace measurement')
        ax.legend()
    for ax in axes[:, 1]:
        ax.set_ylabel('Reconstruction error')
    fig.suptitle('Example trace reconstruction with RNN')
    plt.tight_layout()
    return fig, axes