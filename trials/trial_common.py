import numpy as np

def clamp_model_params(model, val_1, val_2=None):
    if val_2 == None:
        min_param = -val_1
        max_param = val_1
    else:
        min_param = val_1
        max_param = val_2
    for param in model.parameters():
        param.data.clamp_(min_param, max_param)
        
def extract_rv_from_tensor(x):
    return x.detach().cpu().numpy()

def get_model_params_histogram(model, bins=1000):
    params = np.concatenate([extract_rv_from_tensor(param.data.flatten())
                             for param in model.parameters()])
    hist, edges = np.histogram(params, bins=bins)
    return {'histogram': hist,
            'bin_edges': edges}
