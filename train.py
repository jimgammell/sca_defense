from tqdm import tqdm
import numpy as np
import torch

class Results(dict):
    def __init__(self):
        super().__init__()
        self.update = {'generator': {'train_loss': [],
                                   'val_loss': [],
                                   'train_acc': [],
                                   'val_acc': []},
                       'discriminator': {'train_loss': [],
                                         'val_loss': [],
                                         'train_acc': [],
                                         'val_acc': []}}
    def update(self, res):
        for k1 in res.keys():
            for k2 in res[k1].keys():
                self.get(k1)[k2].append(res[k1][k2])

def train_gan_epoch(dataloader, generator, discriminator, generator_loss_fn, discriminator_loss_fn, generator_optimizer, discriminator_optimizer, device):
    generator_losses, generator_accuracies = [], []
    discriminator_losses, discriminator_accuracies = [], []
    for batch in tqdm(dataloader):
        generator_loss, generator_correctness = train_generator_batch(batch, generator, discriminator, generator_loss_fn, generator_optimizer, device)
        generator_losses.append(np.mean(generator_loss))
        generator_accuracies.append(np.mean(generator_correctness))
        discriminator_loss, discriminator_correctness = train_discriminator_batch(batch, generator, discriminator, discriminator_loss_fn, discriminator_optimizer, device)
        discriminator_losses.append(np.mean(discriminator_loss))
        discriminator_accuracies.append(np.mean(discriminator_correctness))
    mean_generator_loss = np.mean(generator_losses)
    mean_generator_acc = np.mean(generator_accuracies)
    mean_discriminator_loss = np.mean(discriminator_losses)
    mean_discriminator_acc = np.mean(discriminator_accuracies)
    return (mean_generator_loss, mean_generator_acc, mean_discriminator_loss, mean_discriminator_acc)

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
            (loss, acc) = train_fn(batch, generator, discriminator, loss_fn, optimizer, device)
        else:
            (loss, acc) = train_fn(batch, generator, discriminator, loss_fn, device)
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

def _eval_discriminator_step(trace, plaintext, key, generator, discriminator, loss_fn):
    discriminator.eval()
    generator.eval()
    with torch.no_grad():
        protected_trace = generator((trace, plaintext, key))
        prediction = discriminator(protected_trace)
        elementwise_loss = loss_fn(prediction, key)
    
    loss_res = elementwise_loss.cpu().numpy()
    pred_res = prediction.cpu().numpy()
    return (loss_res, pred_res)

def _eval_generator_step(trace, plaintext, key, generator, discriminator, loss_fn):
    discriminator.eval()
    generator.eval()
    with torch.no_grad():
        protected_trace = generator((trace, key, plaintext))
        prediction = discriminator(protected_trace)
        elementwise_loss = loss_fn(prediction, key)
        
    loss_res = elementwise_loss.cpu().numpy()
    pred_res = prediction.cpu().numpy()
    return (loss_res, pred_res)

def _train_discriminator_step(trace, plaintext, key, generator, discriminator, loss_fn, optimizer):
    discriminator.train()
    generator.eval()
    with torch.no_grad():
        protected_trace = generator((trace, plaintext, key))
    optimizer.zero_grad()
    prediction = discriminator(protected_trace)
    elementwise_loss = loss_fn(prediction, key)
    loss = torch.mean(elementwise_loss)
    loss.backward()
    optimizer.step()
    
    loss_res = elementwise_loss.detach().cpu().numpy()
    pred_res = prediction.detach().cpu().numpy()
    return (loss_res, pred_res)

def _train_generator_step(trace, plaintext, key, generator, discriminator, loss_fn, optimizer):
    discriminator.eval()
    generator.train()
    optimizer.zero_grad()
    protected_trace = generator((trace, key, plaintext))
    prediction = discriminator(protected_trace)
    elementwise_loss = loss_fn(prediction, key)
    loss = torch.mean(elementwise_loss)
    loss.backward()
    optimizer.step()
    
    loss_res = elementwise_loss.detach().cpu().numpy()
    pred_res = prediction.detach().cpu().numpy()
    return (loss_res, pred_res)

def _execute_step(batch, generator, discriminator, loss_fn, optimizer, device, execute_fn):
    (trace, plaintext, key) = batch
    trace = trace.to(device)
    plaintext = plaintext.to(device)
    key = key.to(device)
    
    (loss_res, pred_res) = execute_fn(trace, plaintext, key, generator, discriminator, loss_fn, optimizer)
    
    correctness_res = np.equal(np.argmax(pred_res, axis=1), key)
    return (loss_res, correctness_res)
