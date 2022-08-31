import numpy as np
import torch
from tqdm import tqdm

from utils import get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))

def train_step(batch, model, optimizer, loss_fn, device):
    model.train()
    x, _, y = batch
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    loss = loss_fn(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

@torch.no_grad()
def eval_step(batch, model, loss_fn, device):
    model.eval()
    x, _, y = batch
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    loss = loss_fn(logits, y).cpu().numpy()
    predictions = np.argmax(logits.cpu().numpy(), axis=-1)
    accuracy = np.mean(np.equal(predictions, y.cpu().numpy()))
    return {'loss': np.mean(loss),
            'accuracy': np.mean(accuracy)}

def train_epoch(dataloader, model, optimizer, loss_fn, device):
    for batch in tqdm(dataloader):
        train_step(batch, model, optimizer, loss_fn, device)

def eval_epoch(dataloader, model, loss_fn, device):
    results = {}
    for batch in tqdm(dataloader):
        results_val = eval_step(batch, model, loss_fn, device)
        for key in results_val.keys():
            if not key in results.keys():
                results.update({key: []})
            results[key].append(results_val[key])
    return results