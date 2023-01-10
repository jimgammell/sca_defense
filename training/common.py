

def unpack_batch(batch, device):
    x, y, _ = batch
    x.to(device)
    y.to(device)
    return x, y

def detach_result(result):
    return result.detach().cpu().numpy()

def run_epoch(step_fn, dataloader, *args, **kwargs):
    results = {}
    for batch in dataloader:
        rv = step_fn(batch, *args, **kwargs)
        for key, item in rv.items():
            if not key in results.keys():
                results[key] = []
            results[key].append(item)
    return results