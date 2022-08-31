

def get_param_count(model):
    num_params = 0
    for _, parameter in model.named_parameters():
        num_params += parameter.numel()
    return num_params