

def train_model(config, checkpoint_dir=None, data_dir=None, get_model=None, device='cuda'):
    
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)