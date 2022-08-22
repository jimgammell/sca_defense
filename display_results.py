from matplotlib import pyplot as plt

def loss_over_time(g_losses_train, g_losses_val, d_losses_train, d_losses_val):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(g_losses_train, color='blue', linestyle='--', label='Train')
    ax[0].plot(g_losses_val, color='blue', linestyle='-', label='Val')
    ax[1].plot(d_losses_train, color='blue', linestyle='--', label='Train')
    ax[1].plot(d_losses_val, color='blue', linestyle='-', label='Val')
    ax[0].set_title('Generator')
    ax[1].set_title('Discriminator')
    for axx in ax:
        axx.set_xlabel('Epoch')
        axx.set_ylabel('Loss')
        axx.legend()
        axx.grid(True)
    fig.suptitle('Loss over time of GAN components')
    plt.tight_layout()
    return fig, ax

def model_output_over_time(obfuscated_signals_train, obfuscated_signals_val, predictions_train, predictions_val):
    def split_trace(trace):
        if len(trace) == 0:
            return trace
        extracted_traces = []
        for idx in range(len(trace[0])):
            extracted_trace = [x[idx] for x in trace]
            extracted_traces.append(extracted_trace)
        return extracted_traces
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    colors = ['blue', 'red']
    for idx, trace in enumerate(split_trace(obfuscated_signals_train)):
        ax[0].plot(trace, color=colors[idx], linestyle='--', label='Train: %d'%(idx))
    for idx, trace in enumerate(split_trace(predictions_train)):
        ax[1].plot(trace, color=colors[idx], linestyle='--', label='Train: %d'%(idx))
    for idx, trace in enumerate(split_trace(obfuscated_signals_val)):
        ax[0].plot(trace, color=colors[idx], linestyle='-', label='Val: %d'%(idx))
    for idx, trace in enumerate(split_trace(predictions_val)):
        ax[1].plot(trace, color=colors[idx], linestyle='-', label='Val: %d'%(idx))
    for idx in range(len(predictions_train[0])):
        ax[1].axhline(idx, color=colors[idx], linestyle='dotted', label='True %d'%(idx))
    ax[0].set_ylabel('Obfuscated data')
    ax[0].set_title('Generator')
    ax[1].set_ylabel('Prediction')
    ax[1].set_title('Discriminator')
    for axx in ax:
        axx.set_xlabel('Epoch')
        axx.legend()
        axx.grid(True)
    fig.suptitle('Outputs over time of GAN components')
    plt.tight_layout()
    return fig, ax