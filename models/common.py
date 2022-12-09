

def get_parameter_count(model, trainable=True, nontrainable=True):
    param_count = 0
    if trainable:
        param_count += sum(p.numel() for p in model.parameters if p.requires_grad)
    if nontrainable:
        param_count += sum(p.numel() for p in model.parameters if not(p.requires_grad))
    return param_count