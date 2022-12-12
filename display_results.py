import torch
from matplotlib import pyplot as plt

@torch.no_grad()
def plot_autoencoder_traces(traces, model, device, fig=None, axes=None):
    traces = traces.to(device)
    autoencoded_traces = model(traces).cpu().numpy()
    traces = traces.cpu().numpy()
    if fig is None:
        assert axes is None
        fig, axes = plt.subplots(traces.shape[0], 2, figsize=(8, 4*traces.shape[0]), sharex=True)
    for idx in range(len(traces)):
        axes[idx, 0].plot(traces[idx], color='blue', label='Original')
        axes[idx, 0].plot(autoencoded_traces[idx], color='red', label='Reconstructed')
        axes[idx, 1].plot(traces[idx]-autoencoded_traces[idx], color='blue', label='Difference')
    return fig, axes