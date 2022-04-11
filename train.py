from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from utils import log_print as print
from dataset import NormTensorMagnitude

class Results(dict):
    def __init__(self):
        super().__init__()
        self.data   = {'generator': {'pretrain_train_loss': [],
                                     'pretrain_val_loss': [],
                                     'train_loss': [],
                                     'val_loss': [],
                                     'train_acc': [],
                                     'val_acc': []},
                       'discriminator': {'pretrain_train_loss': [],
                                         'pretrain_val_loss': [],
                                         'pretrain_train_acc': [],
                                         'pretrain_val_acc': [],
                                         'train_loss': [],
                                         'val_loss': [],
                                         'train_acc': [],
                                         'val_acc': [],
                                         'posttrain_train_loss': [],
                                         'posttrain_val_loss': [],
                                         'posttrain_train_acc': [],
                                         'posttrain_val_acc': []}}
    def update(self, res):
        for k1 in res.keys():
            for k2 in res[k1].keys():
                self.data[k1][k2].append(res[k1][k2])
    def __repr__(self):
        return self.data.__repr__()

def train_gan_epoch(dataloader, generator, discriminator, generator_loss_fn, discriminator_loss_fn, generator_optimizer, discriminator_optimizer, device):
    generator_losses, generator_accuracies = [], []
    discriminator_losses, discriminator_accuracies = [], []
    for batch in tqdm(dataloader):
        generator_loss, generator_correctness = train_generator_step(batch, generator, discriminator, generator_loss_fn, generator_optimizer, device)
        generator_losses.append(np.mean(generator_loss))
        generator_accuracies.append(np.mean(generator_correctness))
        discriminator_loss, discriminator_correctness = train_discriminator_step(batch, generator, discriminator, discriminator_loss_fn, discriminator_optimizer, device)
        discriminator_losses.append(np.mean(discriminator_loss))
        discriminator_accuracies.append(np.mean(discriminator_correctness))
    mean_generator_loss = np.mean(generator_losses)
    mean_generator_acc = np.mean(generator_accuracies)
    mean_discriminator_loss = np.mean(discriminator_losses)
    mean_discriminator_acc = np.mean(discriminator_accuracies)
    return (mean_generator_loss, mean_generator_acc, mean_discriminator_loss, mean_discriminator_acc)

def discriminator_pretrain_eval_epoch(dataloader, discriminator, loss_fn, device):
    losses, accs = [], []
    for batch in tqdm(dataloader):
        (loss, acc) = _eval_pretrain_discriminator_step(batch, discriminator, loss_fn, device)
        losses.append(np.mean(loss))
        accs.append(np.mean(acc))
    mean_loss = np.mean(losses)
    mean_acc = np.mean(accs)
    return (mean_loss, mean_acc)

def _eval_pretrain_discriminator_step(batch, discriminator, loss_fn, device):
    (trace, plaintext), key = batch
    key = key.to(device)
    target = bin_to_int(key, device=device)
    trace = trace.unsqueeze(1).to(device)
    plaintext = plaintext.to(device)
    discriminator.eval()
    with torch.no_grad():
        prediction = discriminator(trace)
        elementwise_loss = loss_fn(prediction, target)
    loss_res = elementwise_loss.detach().cpu().numpy()
    pred_res = prediction.detach().cpu().numpy()
    correctness_res = np.equal(np.argmax(pred_res, axis=-1), target.cpu().numpy())
    return (loss_res, correctness_res)

def discriminator_pretrain_epoch(dataloader, discriminator, loss_fn, optimizer, device):
    losses, accs = [], []
    for batch in tqdm(dataloader):
        (loss, acc) = _pretrain_discriminator_step(batch, discriminator, loss_fn, optimizer, device)
        losses.append(np.mean(loss))
        accs.append(np.mean(acc))
    mean_loss = np.mean(losses)
    mean_acc = np.mean(accs)
    return (mean_loss, mean_acc)

def _pretrain_discriminator_step(batch, discriminator, loss_fn, optimizer, device):
    (trace, plaintext), key = batch
    key = key.to(device)
    target = bin_to_int(key, device=device)
    trace = trace.unsqueeze(1).to(device)
    plaintext = plaintext.to(device)
    discriminator.train()
    optimizer.zero_grad()
    prediction = discriminator(trace)
    elementwise_loss = loss_fn(prediction, target)
    loss = torch.mean(elementwise_loss)
    loss.backward()
    optimizer.step()
    
    loss_res = elementwise_loss.detach().cpu().numpy()
    pred_res = prediction.detach().cpu().numpy()
    correctness_res = np.equal(np.argmax(pred_res, axis=-1), target.cpu().numpy())
    return (loss_res, correctness_res)

def generator_pretrain_autoencoder_epoch(dataloader, generator, optimizer, device):
    loss_fn = nn.MSELoss()
    res = execute_generator_pretrain_epoch(dataloader, generator, loss_fn, optimizer, device, _step_generator_autoencoder)
    return res

def generator_eval_autoencoder_epoch(dataloader, generator, device):
    loss_fn = nn.MSELoss()
    optimizer = None
    losses = []
    for batch in tqdm(dataloader):
        loss = _eval_generator_autoencoder(batch, generator, loss_fn, device)
        losses.append(np.mean(loss))
    mean_loss = np.mean(losses)
    return mean_loss

def execute_generator_pretrain_epoch(dataloader, generator, loss_fn, optimizer, device, execute_fn):
    losses = []
    for batch in tqdm(dataloader):
        if optimizer != None:
            loss = execute_fn(batch, generator, loss_fn, optimizer, device)
        else:
            loss = execute_fn(batch, generator, loss_fn, device)
        losses.append(np.mean(loss))
    mean_loss = np.mean(losses)
    return mean_loss

def _eval_generator_autoencoder(batch, generator, loss_fn, device):
    (trace, plaintext), key = batch
    trace = trace.to(device)
    plaintext = plaintext.to(device)
    key = key.to(device)
    generator.eval()
    with torch.no_grad():
        protected_trace = generator((trace, key, plaintext)).squeeze()
        elementwise_loss = loss_fn(trace, protected_trace)
    return elementwise_loss.cpu().numpy()

def _step_generator_autoencoder(batch, generator, loss_fn, optimizer, device):
    (trace, plaintext), key = batch
    trace = trace.to(device)
    plaintext = plaintext.to(device)
    key = key.to(device)
    generator.train()
    optimizer.zero_grad()
    protected_trace = generator((trace, key, plaintext)).squeeze()
    elementwise_loss = loss_fn(trace, protected_trace)
    loss = torch.mean(elementwise_loss)
    loss.backward()
    optimizer.step()
    
    loss_res = elementwise_loss.detach().cpu().numpy()
    return loss_res

def train_generator_epoch(dataloader, generator, discriminator, loss_fn, optimizer, device):
    res = _execute_single_model_epoch(dataloader, generator, discriminator, loss_fn, optimizer, device, train_generator_step)
    return res

def train_discriminator_epoch(dataloader, generator, discriminator, loss_fn, optimizer, device):
    res = _execute_single_model_epoch(dataloader, generator, discriminator, loss_fn, optimizer, device, train_discriminator_step)
    return res

def eval_generator_epoch(dataloader, generator, discriminator, loss_fn, device):
    res = _execute_single_model_epoch(dataloader, generator, discriminator, loss_fn, None, device, eval_generator_step)
    return res

def eval_discriminator_epoch(dataloader, generator, discriminator, loss_fn, device):
    res = _execute_single_model_epoch(dataloader, generator, discriminator, loss_fn, None, device, eval_discriminator_step)
    return res

def _execute_single_model_epoch(dataloader, generator, discriminator, loss_fn, optimizer, device, execute_fn):
    losses, accuracies = [], []
    for batch in tqdm(dataloader):
        if optimizer != None:
            (loss, acc) = execute_fn(batch, generator, discriminator, loss_fn, optimizer, device)
        else:
            (loss, acc) = execute_fn(batch, generator, discriminator, loss_fn, device)
        losses.append(np.mean(loss))
        accuracies.append(np.mean(acc))
    mean_loss = np.mean(losses)
    mean_acc = np.mean(accuracies)
    return (mean_loss, mean_acc)

def eval_discriminator_step(batch, generator, discriminator, loss_fn, device):
    res = _execute_step(batch, generator, discriminator, loss_fn, None, device, _eval_discriminator_step)
    return res

def eval_generator_step(batch, generator, discriminator, loss_fn, device):
    res = _execute_step(batch, generator, discriminator, loss_fn, None, device, _eval_generator_step)
    return res

def train_discriminator_step(batch, generator, discriminator, loss_fn, optimizer, device):
    res = _execute_step(batch, generator, discriminator, loss_fn, optimizer, device, _train_discriminator_step)
    return res

def train_generator_step(batch, generator, discriminator, loss_fn, optimizer, device):
    res = _execute_step(batch, generator, discriminator, loss_fn, optimizer, device, _train_generator_step)
    return res

def _eval_discriminator_step(trace, plaintext, key, target, generator, discriminator, loss_fn):
    discriminator.eval()
    generator.eval()
    with torch.no_grad():
        protected_trace = generator((trace, plaintext, key))
        prediction = discriminator(protected_trace)
        elementwise_loss = loss_fn(prediction, target)
    
    loss_res = elementwise_loss.cpu().numpy()
    pred_res = prediction.cpu().numpy()
    return (loss_res, pred_res)

def _eval_generator_step(trace, plaintext, key, target, generator, discriminator, loss_fn):
    discriminator.eval()
    generator.eval()
    with torch.no_grad():
        protected_trace = generator((trace, key, plaintext))
        prediction = discriminator(protected_trace)
        elementwise_loss = -loss_fn(prediction, target)
        
    loss_res = elementwise_loss.cpu().numpy()
    pred_res = prediction.cpu().numpy()
    return (loss_res, pred_res)

def _train_discriminator_step(trace, plaintext, key, target, generator, discriminator, loss_fn, optimizer):
    discriminator.train()
    generator.train()
    with torch.no_grad():
        protected_trace = generator((trace, plaintext, key))
    optimizer.zero_grad()
    prediction = discriminator(protected_trace)
    elementwise_loss = loss_fn(prediction, target)
    loss = torch.mean(elementwise_loss)
    loss.backward()
    optimizer.step()
    
    loss_res = elementwise_loss.detach().cpu().numpy()
    pred_res = prediction.detach().cpu().numpy()
    return (loss_res, pred_res)

def _train_generator_step(trace, plaintext, key, target, generator, discriminator, loss_fn, optimizer):
    discriminator.train()
    generator.train()
    optimizer.zero_grad()
    protected_trace = generator((trace, key, plaintext))
    prediction = discriminator(protected_trace)
    elementwise_loss = -loss_fn(prediction, target)
    loss = torch.mean(elementwise_loss)
    loss.backward()
    optimizer.step()
    
    loss_res = elementwise_loss.detach().cpu().numpy()
    pred_res = prediction.detach().cpu().numpy()
    return (loss_res, pred_res)

def bin_to_int(x, device, bits=8, classes=256):
    bases = torch.tensor([2**n for n in range(bits-1, -1, -1)], dtype=torch.int, device=device)
    x_int = (x*bases).sum(dim=-1).long()
    return x_int
                               
def _execute_step(batch, generator, discriminator, loss_fn, optimizer, device, execute_fn):
    (trace, plaintext), key = batch
    trace = trace.to(device)
    plaintext = plaintext.to(device)
    key = key.to(device)
    target = bin_to_int(key, device=device)
    
    if optimizer != None:
        (loss_res, pred_res) = execute_fn(trace, plaintext, key, target, generator, discriminator, loss_fn, optimizer)
    else:
        (loss_res, pred_res) = execute_fn(trace, plaintext, key, target, generator, discriminator, loss_fn)
    
    correctness_res = np.equal(np.argmax(pred_res, axis=1), target.cpu().numpy())
    return (loss_res, correctness_res)
