import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from training.common import *

@torch.no_grad()
def val(x):
    if not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()

@torch.no_grad()
def int_to_binary(x):
    assert x.dtype == torch.long
    out = torch.zeros(x.size(0), 8, dtype=torch.float, device=x.device)
    for n in range(7, -1, -1):
        high = torch.ge(x, 2**n)
        out[:, n] = high
        x -= high*2**n
    return out

@torch.no_grad()
def bin_acc(x, y):
    x, y = val(x), val(y)
    return np.mean(np.equal(x>0, y))

@torch.no_grad()
def calculate_mean_accuracy(dataloader, gen, disc, device):
    gen.eval()
    disc.eval()
    acc_orig_labels = acc_rec_labels = 0.0
    for bidx, batch in enumerate(dataloader):
        trace, labels = batch
        trace, labels = trace.to(device), {key: item.to(device) for key, item in labels.items()}
        labels_rec = {key: torch.randint_like(item, 256) for key, item in labels.items()}
        binary_labels_rec = {key: int_to_binary(item) for key, item in labels_rec.items()}
        trace_rec = gen(trace, binary_labels_rec)
        features_rec = disc(trace_rec)
        logits_rec = disc.classify_leakage(features_rec)
        batch_acc_orig_labels = torch.cat([acc(disc_logits_rec[key], labels[key]) for key in labels.keys()], dim=-1).mean()
        batch_acc_rec_labels = torch.cat([acc(disc_logits_rec[key], labels_rec[key]) for key in labels_rec.keys()], dim=-1).mean()
        acc_orig_labels += batch_acc_orig_labels
        acc_rec_labels += batch_acc_rec_labels
    acc_orig_labels /= len(dataloader)
    acc_rec_labels /= len(dataloader)
    return acc_orig_labels, acc_rec_labels

def mean_val(d):
    return torch.mean(torch.tensor(list(d.values())))

def apply_elementwise(fn, logits, labels):
    rv = {}
    for key in labels.keys():
        attack_pt, byte = key.split('__')
        for head_key in logits.keys():
            _, attack_pt_, byte_ = head_key.split('__')
            if attack_pt_ == attack_pt and byte_ == byte:
                break
        rv[key] = fn(logits[head_key], labels[key])
    return rv

def leakage_loss(logits, labels):
    return apply_elementwise(lambda x, y: nn.functional.binary_cross_entropy(torch.sigmoid(x), y), logits, labels)

def train_step_cyclegan(batch, gen, gen_opt, disc, disc_opt, device,
                        gen_opt_scheduler=None, disc_opt_scheduler=None,
                        l1_rec_coefficient=0.0, gen_classification_coefficient=1.0,
                        return_example=False, return_weight_norms=False, return_grad_norms=False,
                        average_deviation_penalty=0.0, train_gen=True, train_disc=True, **kwargs):
    trace, labels = batch
    trace, labels = trace.to(device), {key: item.to(device) for key, item in labels.items()}
    labels_rec = {key: torch.randint_like(item, 256) for key, item in labels.items()}
    labels = {key: int_to_binary(item) for key, item in labels.items()}
    labels_rec = {key: int_to_binary(item) for key, item in labels_rec.items()}
    gen.train()
    disc.train()
    rv = {}
    
    if train_disc:
        with torch.no_grad():
            trace_rec = gen(trace, labels_rec)
        disc_features_orig = disc.extract_features(trace)
        disc_features_rec = disc.extract_features(trace_rec)
        
        disc_logits_orig_leakage = disc.classify_leakage(disc_features_orig)
        disc_logits_rec_leakage = disc.classify_leakage(disc_features_rec)
        disc_loss_orig_leakage = leakage_loss(disc_logits_orig_leakage, labels)
        disc_loss_rec_leakage = leakage_loss(disc_logits_rec_leakage, labels)
        disc_loss_leakage = 0.5*mean_val(disc_loss_orig_leakage) + 0.5*mean_val(disc_loss_rec_leakage)
        
        disc_logits_orig_realism = disc.classify_realism(disc_features_orig, labels)
        disc_logits_rec_realism = disc.classify_realism(disc_features_rec, labels_rec)
        disc_loss_orig_realism = hinge_loss(disc_logits_orig_realism, 1)
        disc_loss_rec_realism = hinge_loss(disc_logits_rec_realism, -1)
        disc_loss_realism = 0.5*disc_loss_orig_realism + 0.5*disc_loss_rec_realism
        
        disc_loss = 0.5*disc_loss_leakage + 0.5*disc_loss_realism
        disc_opt.zero_grad(set_to_none=True)
        disc_loss.backward()
        if return_grad_norms:
            rv.update({'disc_grad_norm': get_grad_norm(disc)})
        disc_opt.step()
        
        rv.update({
            'disc_loss_realism': val(disc_loss_realism),
            'disc_loss_leakage': val(disc_loss_leakage),
            'disc_loss_orig_leakage': val(disc_loss_orig_leakage),
            'disc_loss_rec_leakage': val(disc_loss_rec_leakage),
            'disc_loss': val(disc_loss),
            'disc_acc_realism': 0.5*hinge_acc(disc_logits_orig_realism, 1) + 0.5*hinge_acc(disc_logits_rec_realism, -1),
            'disc_acc_orig_leakage': apply_elementwise(bin_acc, disc_logits_orig_leakage, labels),
            'disc_acc_rec_leakage': apply_elementwise(bin_acc, disc_logits_rec_leakage, labels)
        })
        rv.update({'disc_acc_leakage': 0.5*mean_val(rv['disc_acc_orig_leakage']) + 0.5*mean_val(rv['disc_acc_rec_leakage'])})
    
    if train_gen:
        trace_rec = gen(trace, labels_rec)
        disc_features_rec = disc.extract_features(trace_rec)
        
        disc_logits_rec_leakage = disc.classify_leakage(disc_features_rec)
        gen_loss_rec_leakage = leakage_loss(disc_logits_rec_leakage, labels_rec)
        gen_loss_leakage = mean_val(gen_loss_rec_leakage)
        
        disc_logits_rec_realism = disc.classify_realism(disc_features_rec, labels_rec)
        gen_loss_realism = -disc_logits_rec_realism.mean()
        
        if l1_rec_coefficient > 0:
            trace_crec = gen(trace_rec, labels)
            gen_l1_loss = nn.functional.l1_loss(trace, trace_crec)
        
        gen_loss = gen_loss_realism + gen_classification_coefficient*gen_loss_leakage + l1_rec_coefficient*gen_l1_loss
        if average_deviation_penalty != 0.0:
            gen_avg_departure_penalty = gen.get_avg_departure_penalty()
            gen_loss += average_deviation_penalty*gen_avg_departure_penalty
        gen_opt.zero_grad(set_to_none=True)
        gen_loss.backward()
        if return_grad_norms:
            rv.update({'gen_grad_norm': get_grad_norm(gen)})
        gen_opt.step()
        if average_deviation_penalty != 0.0:
            gen.update_avg()
        elif average_deviation_penalty != 0.0:
            gen.reset_avg()
        
        rv.update({
            'gen_loss_realism': val(gen_loss_realism),
            'gen_loss_leakage': val(gen_loss_leakage),
            'gen_loss_rec_leakage': {key: val(item) for key, item in gen_loss_rec_leakage.items()},
            'gen_l1_loss': val(gen_l1_loss),
            'gen_avg_departure_loss': val(gen_avg_departure_penalty),
            'gen_loss': val(gen_loss),
            'gen_acc_realism': hinge_acc(disc_logits_rec_realism, 1),
            'gen_acc_rec_leakage': apply_elementwise(bin_acc, disc_logits_rec_leakage, labels_rec),
            'reconstruction_diff_l1': val((trace-trace_rec).norm(p=1))
        })
        rv.update({'gen_acc_leakage': mean_val(rv['gen_acc_rec_leakage'])})
    if return_example:
        rv.update({
            'orig_example': val(trace),
            'rec_example': val(trace_rec)
        })
        if l1_rec_coefficient > 0:
            rv.update({'crec_example': val(trace_crec)})
    if return_weight_norms:
        rv.update({
            'disc_weight_norm': get_weight_norm(disc),
            'gen_weight_norm': get_weight_norm(gen)
        })
    if gen_opt_scheduler is not None:
        gen_opt_scheduler.step()
    if disc_opt_scheduler is not None:
        disc_opt_scheduler.step()
    return rv

@torch.no_grad()
def eval_step_cyclegan(batch, gen, disc, device, l1_rec_coefficient=0.0, gen_classification_coefficient=1.0, return_example=False, **kwargs):
    trace, labels = batch
    trace, labels = trace.to(device), {key: item.to(device) for key, item in labels.items()}
    labels_rec = {key: torch.randint_like(item, 256) for key, item in labels.items()}
    labels = {key: int_to_binary(item) for key, item in labels.items()}
    labels_rec = {key: int_to_binary(item) for key, item in labels_rec.items()}
    gen.eval()
    disc.eval()
    rv = {}
    
    trace_rec = gen(trace, labels_rec)
    disc_features_orig = disc.extract_features(trace)
    disc_features_rec = disc.extract_features(trace_rec)
    
    disc_logits_orig_leakage = disc.classify_leakage(disc_features_orig)
    disc_logits_rec_leakage = disc.classify_leakage(disc_features_rec)
    disc_loss_orig_leakage = leakage_loss(disc_logits_orig_leakage, labels)
    disc_loss_rec_leakage = leakage_loss(disc_logits_rec_leakage, labels)
    disc_loss_leakage = 0.5*mean_val(disc_loss_orig_leakage) + 0.5*mean_val(disc_loss_rec_leakage)
    
    disc_logits_orig_realism = disc.classify_realism(disc_features_orig, labels)
    disc_logits_rec_realism = disc.classify_realism(disc_features_rec, labels_rec)
    disc_loss_orig_realism = hinge_loss(disc_logits_orig_realism, 1)
    disc_loss_rec_realism = hinge_loss(disc_logits_rec_realism, -1)
    disc_loss_realism = 0.5*disc_loss_orig_realism, 0.5*disc_loss_rec_realism
    
    disc_loss = 0.5*disc_loss_leakage + 0.5*disc_loss_realism
    
    gen_loss_rec_leakage = leakage_loss(disc_logits_rec_leakage, labels_rec)
    gen_loss_leakage = mean_val(gen_loss_rec_leakage)
    
    gen_loss_realism = -disc_logits_rec_realism.mean()
    
    if l1_rec_coefficient > 0:
        trace_crec = gen(trace_rec, labels)
        gen_l1_loss = nn.functional.l1_loss(trace, trace_crec)
        
    gen_loss = gen_loss_realism + gen_classification_coefficient*gen_loss_leakage + l1_rec_coefficient*gen_l1_loss
    
    rv.update({
        'disc_loss_realism': val(disc_loss_realism),
        'disc_loss_leakage': val(disc_loss_leakage),
        'disc_loss_orig_leakage': val(disc_loss_orig_leakage),
        'disc_loss_rec_leakage': val(disc_loss_rec_leakage),
        'disc_loss': val(disc_loss),
        'disc_acc_realism': 0.5*hinge_acc(disc_logits_orig_realism, 1) + 0.5*hinge_acc(disc_logits_rec_realism, -1),
        'disc_acc_orig_leakage': apply_elementwise(bin_acc, disc_logits_orig_leakage, labels),
        'disc_acc_rec_leakage': apply_elementwise(bin_acc, disc_logits_rec_leakage, labels)
    })
    rv.update({'disc_acc_leakage': 0.5*mean_val(rv['disc_acc_orig_leakage']) + 0.5*mean_val(rv['disc_acc_rec_leakage'])})
    rv.update({
        'gen_loss_realism': val(gen_loss_realism),
        'gen_loss_leakage': val(gen_loss_leakage),
        'gen_loss_rec_leakage': {key: val(item) for key, item in gen_loss_reg_leakage.items()},
        'gen_l1_loss': val(gen_l1_loss),
        'gen_avg_departure_loss': val(gen_avg_departure_penalty),
        'gen_loss': val(gen_loss),
        'gen_acc_realism': hinge_acc(disc_logits_rec_realism, 1),
        'gen_acc_rec_leakage': apply_elementwise(bin_acc, disc_logits_rec_leakage, labels_rec),
        'reconstruction_diff_l1': val((trace-trace-rec).norm(p=1))
    })
    rv.update({'gen_acc_leakage': mean_val(rv['gen_acc_rec_leakage'])})
    if return_example:
        rv.update({
            'orig_example': val(trace),
            'rec_example': val(trace_rec)
        })
    if l1_rec_coefficient > 0:
        rv.update({'crec_example': val(trace_crec)})
    return rv

def train_step(batch, model, optimizer, lr_scheduler, device):
    rv = {}
    model.train()
    trace, labels = batch
    trace, labels = trace.to(device), {key: item.to(device) for key, item in labels.items()}
    logits = model(trace)
    
    loss = 0.0
    for head_name, head_logits in logits.items():
        tr, tap, tb = head_name.split('__')
        target = labels['{}__{}'.format(tap, tb)]
        if tr == 'bits':
            target = int_to_binary(target)
            loss_h = nn.functional.binary_cross_entropy(torch.sigmoid(head_logits), target.to(torch.float))
            acc_h = bin_acc(head_logits, target)
        elif tr == 'bytes':
            loss_h = nn.functional.cross_entropy(head_logits, target)
            acc_h = acc(head_logits, target)
        else:
            raise NotImplementedError
        loss += loss_h
        rv[head_name+'__loss'] = loss_h.detach().cpu().numpy()
        rv[head_name+'__acc'] = acc_h
    loss /= len(logits)
    rv['total_loss'] = val(loss)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    
    return rv

def eval_step(batch, model, device):
    rv = {}
    model.eval()
    trace, labels = batch
    trace, labels = trace.to(device), {key: item.to(device) for key, item in labels.items()}
    logits = model(trace)
    
    loss = 0.0
    for head_name, head_logits in logits.items():
        tr, tap, tb = head_name.split('__')
        target = labels['{}__{}'.format(tap, tb)]
        if tr == 'bits':
            target = int_to_binary(target)
            loss_h = nn.functional.binary_cross_entropy(torch.sigmoid(head_logits), target.to(torch.float))
            acc_h = bin_acc(head_logits, target)
        elif tr == 'bytes':
            loss_h = nn.functional.cross_entropy(head_logits, target)
            acc_h = acc(head_logits, target)
        else:
            raise NotImplementedError
        loss += loss_h
        rv[head_name+'__loss'] = loss_h.detach().cpu().numpy()
        rv[head_name+'__acc'] = acc_h
    loss /= len(logits)
    rv['total_loss'] = val(loss)
    
    return rv

def run_epoch(dataloader, step_fn, *step_args):
    rv = {}
    for batch in tqdm(dataloader):
        batch_rv = step_fn(batch, *step_args)
        for key, item in batch_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    for key, item in rv.items():
        rv[key] = np.mean(item)
    return rv

def train_epoch(dataloader, model, optimizer, lr_scheduler, device):
    return run_epoch(dataloader, train_step, model, optimizer, lr_scheduler, device)
def eval_epoch(dataloader, model, device):
    return run_epoch(dataloader, eval_step, model, device)

def train_cyclegan_epoch(dataloader, *step_args, return_example_idx=None, disc_steps_per_gen_step=1.0, **step_kwargs):
    rv = {}
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx) == int else return_example_idx
    disc_steps = gen_steps = 0
    for bidx, batch in enumerate(dataloader):
        if disc_steps_per_gen_step > 1:
            disc_steps += 1
            train_disc = True
            train_gen = disc_steps >= disc_steps_per_gen_step
            if train_gen:
                disc_steps -= disc_steps_per_gen_step
        elif 1/disc_steps_per_gen_step > 1:
            gen_steps += 1
            train_gen = True
            train_disc = gen_steps >= 1/disc_steps_per_gen_step
            if gen_steps >= 1/disc_steps_per_gen_step:
                gen_steps -= 1/disc_steps_per_gen_step
        else:
            train_disc = train_gen = True
        step_rv = train_step_cyclegan(
            batch, *step_args,
            return_example=bidx in re_indices, train_gen=train_gen, train_disc=train_disc, **step_kwargs
        )
        for key, item in step_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    for key, item in rv.items():
        if not 'example' in key:
            rv[key] = np.nanmean(item)
    return rv

def eval_cyclegan_epoch(dataloader, *step_args, return_example_idx=None, posttrain=False,
                        leakage_eval_disc=None, **step_kwargs):
    rv = {}
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx) == int else return_example_idx
    for bidx, batch in enumerate(dataloader):
        step_rv = eval_step_cyclegan(batch, *step_args, return_example_idx=bidx in re_indices, **step_kwargs)
        for key, item in step_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    if leakage_eval_disc is not None:
        gen = step_args[0]
        device = step_args[-1]
        acc_orig_labels, acc_rec_labels = calculate_mean_accuracy(dataloader, gen, leakage_eval_disc, device)
        rv['acc_leakage_orig_labels'] = acc_orig_labels
        rv['acc_leakage_rec_labels'] = acc_rec_labels
    for key, item in rv.items():
        if not 'example' in key:
            rv[key] = np.nanmean(item)
    return rv