import numpy as np
import torch
from torch import nn, optim

# Things to try:
#   Use only one discriminator feature extractor for all functionalities

def val(tensor):
    try:
        return tensor.detach().cpu().numpy()
    except:
        return np.nan

def acc(logits, labels):
    logits, labels = val(logits), val(labels)
    acc = np.mean(np.equal(np.argmax(logits, axis=-1), labels))
    return acc

def to_uint8(x):
    x = 255.0*(0.5*x+0.5)
    x = x.to(torch.uint8)
    x = x.to(torch.float)
    x = 2.0*(x/255.0)-1.0
    return x

def get_weight_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.detach().norm(2)
        total_norm += param_norm.item()**2
    total_norm = total_norm**0.5
    return total_norm

def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().norm(2)
            total_norm += param_norm.item()**2
    total_norm = total_norm**0.5
    return total_norm

def hinge_loss(logits, y):
    return nn.functional.relu(1-y*logits).mean()

def multiclass_hinge_loss(logits, y):
    output_y = logits[torch.arange(0, y.size(0)), y.data].view(-1, 1)
    loss = logits - output_y + 1.0
    loss[torch.arange(0, y.size(0)), y.data] = 0
    loss[loss<0] = 0
    loss = loss.mean()
    return loss

def inv_multiclass_hinge_loss(logits, y):
    output_y = logits[torch.arange(0, y.size(0)), y.data].view(-1, 1)
    loss = -logits + output_y + 1.0
    loss[torch.arange(0, y.size(0)), y.data] = 0
    loss[loss<0] = 0
    loss = loss.mean()
    return loss

def stats_l1_loss(logits, y):
    loss = 0.0
    mean = logits.mean(dim=0, keepdim=True)
    for yy in torch.unique(y):
        loss += (logits[y==yy]-mean).norm(p=1)
    loss /= len(logits)
    return loss
    
def get_invariance_penalty(f1, f2, y):
    invariance_penalty = 0.0
    f = torch.cat((f1, f2), dim=0)
    yy = torch.cat((y, y), dim=0)
    for y_A in torch.unique(y):
        for y_B in torch.unique(y):
            invariance_penalty = invariance_penalty + (f[yy==y_A].mean(0)-f[yy==y_B].mean(0)).norm(p=1)
            invariance_penalty = invariance_penalty + (f[yy==y_A].std(0)-f[yy==y_B].std(0)).norm(p=1)
    return invariance_penalty

def confusion_loss(logits, y):
    model_log_probs = nn.functional.log_softmax(logits, dim=-1)
    baseline_probs = nn.functional.softmax(torch.zeros_like(logits), dim=-1)
    baseline_log_probs = nn.functional.log_softmax(torch.zeros_like(logits), dim=-1)
    kl_div = (baseline_probs*(baseline_log_probs-model_log_probs)).sum(dim=-1).mean()
    return kl_div

def compute_optimal_example(example, disc, gen, eps=1e-5, max_iter=100, opt=optim.Adam, opt_kwargs={'lr': 1e-2}):
    confusing_example = gen(example).detach()
    confusing_example.requires_grad = True
    opt = opt([confusing_example], **opt_kwargs)
    criterion = confusion_loss
    disc_output_real = disc(example).detach()
    prev_loss = np.inf
    for current_iter in range(max_iter):
        disc_output_fake = disc(confusing_example)
        loss = criterion(disc_output_fake, disc_output_real)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if torch.abs(loss.detach()-prev_loss) < eps:
            break
        prev_loss = loss.detach()
    return confusing_example.detach(), loss.detach().cpu().numpy(), current_iter

def get_std(x):
    return (((x - x.mean(dim=0, keepdim=True))**2).mean(dim=0, keepdim=True))**0.5

def whiten_features(features, y, sub_realism_mean=False, div_realism_std=False):
    if sub_realism_mean:
        mn = features.mean(dim=0, keepdim=True)
        for yy in torch.unique(y):
            mn_yy = features[y==yy].mean(dim=0, keepdim=True)
            features[y==yy] = features[y==yy] - mn_yy + mn
    if div_realism_std:
        std = get_std(features)
        for yy in torch.unique(y):
            std_yy = get_std(features[y==yy])
            features[y==yy] = features[y==yy] * std / std_yy
    return features

def feature_whitening_penalty(features, y):
    loss = 0.0
    mn = features.mean(dim=0, keepdim=True)
    std = get_std(features)
    for yy in torch.unique(y):
        mn_yy = features[y==yy].mean(dim=0, keepdim=True)
        std_yy = get_std(features[y==yy])
        loss += (mn_yy - mn).norm(p=1)
        loss += (std_yy - std).norm(p=1)
    loss /= len(features)
    return loss

def train_single_step(batch, model, opt, loss_fn, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    model.train()
    logits = model(x)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    
    rv = {
        'loss': val(loss),
        'acc': acc(logits, y)
    }
    return rv

@torch.no_grad()
def eval_single_step(batch, model, loss_fn, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    model.eval()
    logits = model(x)
    loss = loss_fn(logits, y)
    rv = {
        'loss': val(loss),
        'acc': acc(logits, y)
    }
    return rv

def train_step(batch, gen, gen_opt, disc, disc_opt, device, project_gen_updates=False, leakage_batch=None,
               return_example=False, return_weight_norms=False, return_grad_norms=False,
               sub_realism_mean=False, div_realism_std=False,
               gen_leakage_coefficient=0.5, disc_invariance_coefficient=0.5,
               train_gen=True, train_disc=True, 
               disc_leakage_loss=multiclass_hinge_loss, gen_leakage_loss=stats_l1_loss,
               train_leakage_disc_on_orig_samples=False,
               whitening_averaging_coefficient=0.0,
               gen_swa=None, disc_swa=None,
               eval_gen=None, eval_gen_beta=None):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    disc.train()
    gen.train()
    gen_tr = gen_swa if gen_swa is not None else gen
    disc_tr = disc_swa if disc_swa is not None else disc
    gen_tr.train()
    disc_tr.train()
    rv = {}
    
    # discriminator update
    if train_disc:
        x_rec = gen_tr(x).detach()
        features_real = disc.realism_analyzer.extract_features(torch.cat((x, x), dim=1))
        features_real = whiten_features(features_real, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std)
        logits_real = disc.realism_analyzer.classify_features(features_real)
        features_fake = disc.realism_analyzer.extract_features(torch.cat((x, x_rec), dim=1))
        features_fake = whiten_features(features_fake, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std)
        logits_fake = disc.realism_analyzer.classify_features(features_fake)
        disc_loss_realism = 0.5*hinge_loss(logits_real, 1) + 0.5*hinge_loss(logits_fake, -1)
        #features_real = disc.realism_analyzer.extract_features(x)
        #features_real = whiten_features(features_real, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std)
        #logits_real = disc.realism_analyzer.classify_features(torch.cat((features_real, features_real), dim=1))
        #features_fake = disc.realism_analyzer.extract_features(x_rec)
        #features_fake = whiten_features(features_fake, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std) ##
        #logits_fake = disc.realism_analyzer.classify_features(torch.cat((features_real, features_fake), dim=1))
        #disc_loss_realism = 0.5*hinge_loss(logits_real, 1) + 0.5*hinge_loss(logits_fake, -1)
        if leakage_batch is not None:
            x_lk, y_lk, _ = leakage_batch
            x_lk, y_lk = x_lk.to(device), y_lk.to(device)
            x_rec = gen_tr(x_lk).detach()
        else:
            x_lk = x
            y_lk = y
        if train_leakage_disc_on_orig_samples:
            leakage_real = disc.assess_leakage(x_lk)
            disc_loss_leakage = disc_leakage_loss(leakage_real, y_lk)
        else:
            leakage_fake = disc.assess_leakage(x_rec)
            disc_loss_leakage = disc_leakage_loss(leakage_fake, y_lk)
        disc_loss = disc_loss_realism + disc_loss_leakage
        disc_opt.zero_grad()
        disc_loss.backward()
        if return_grad_norms:
            rv.update({'disc_grad_norm': get_grad_norm(disc)})
        disc_opt.step()
        if disc_swa is not None:
            disc_swa.update_parameters(disc)
    else:
        disc_loss = disc_loss_realism = disc_loss_leakage = None
    
    # generator update
    if train_gen:
        x_rec = gen(x)
        features_fake = disc_tr.realism_analyzer.extract_features(torch.cat((x, x_rec), dim=1))
        features_fake = whiten_features(features_fake, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std)
        logits_fake = disc_tr.realism_analyzer.classify_features(features_fake)
        #features_real = disc.realism_analyzer.extract_features(x)
        #features_real = whiten_features(features_real, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std)
        #features_fake = disc.realism_analyzer.extract_features(x_rec)
        #features_fake = whiten_features(features_fake, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std) ##
        #logits_fake = disc.realism_analyzer.classify_features(torch.cat((features_real, features_fake), dim=1))
        gen_loss_realism = -logits_fake.mean()
        leakage = disc_tr.assess_leakage(x_rec)
        gen_loss_leakage = gen_leakage_loss(leakage, y)
        gen_loss = gen_leakage_coefficient*gen_loss_leakage + (1-gen_leakage_coefficient)*gen_loss_realism
        if project_gen_updates: # and leakage loss > threshold
            dot = lambda x, y: (x*y).sum()
            gen_opt.zero_grad()
            gen_loss_leakage.backward(retain_graph=True)
            gen_leakage_gradients = [param.grad.clone() for param in gen.parameters()]
            gen_opt.zero_grad()
            gen_loss_realism.backward()
            for leakage_grad, param in zip(gen_leakage_gradients, gen.parameters()):
                realism_grad = param.grad
                projected_grad = realism_grad - leakage_grad*dot(leakage_grad, realism_grad)/(dot(leakage_grad, leakage_grad)+1e-12)
                projected_grad = 0.5*projected_grad + 0.5*leakage_grad
                param.grad = projected_grad
        else:
            gen_opt.zero_grad()
            gen_loss.backward()
        if return_grad_norms:
            rv.update({'gen_grad_norm': get_grad_norm(gen)})
        gen_opt.step()
        if gen_swa is not None:
            gen_swa.update_parameters(gen)
        if (eval_gen is not None) and (eval_gen_beta is not None):
            with torch.no_grad():
                sd_train, sd_eval = gen.state_dict(), eval_gen.state_dict()
                for key in sd_train.keys():
                    if sd_train[key].dtype != torch.float:
                        continue
                    assert key in sd_eval.keys()
                    sd_eval[key] = eval_gen_beta*sd_eval[key] + (1-eval_gen_beta)*sd_train[key]
                eval_gen.load_state_dict(sd_eval)
    else:
        gen_loss = gen_loss_realism = gen_loss_leakage = None
    
    rv.update({
        'disc_loss_realism': val(disc_loss_realism),
        'disc_loss_leakage': val(disc_loss_leakage),
        'disc_loss': val(disc_loss),
        'gen_loss_realism': val(gen_loss_realism),
        'gen_loss_leakage': val(gen_loss_leakage),
        'gen_loss': val(gen_loss)
    })
    if return_example:
        rv.update({
            'orig_example': 0.5*val(to_uint8(x))+0.5,
            'rec_example': 0.5*val(to_uint8(x_rec))+0.5
        })
    if return_weight_norms:
        rv.update({
            'disc_weight_norm': get_weight_norm(disc),
            'gen_weight_norm': get_weight_norm(gen)
        })
    return rv

@torch.no_grad()
def eval_step(batch, gen, disc, device,
              return_example=False, sub_realism_mean=False, div_realism_std=False,
              gen_leakage_coefficient=0.5, disc_invariance_coefficient=0.5,
              train_leakage_disc_on_orig_samples=False,
              whitening_averaging_coefficient=0.0,
              disc_swa=None, gen_swa=None,
              disc_leakage_loss=multiclass_hinge_loss, gen_leakage_loss=stats_l1_loss):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    if disc_swa is not None:
        disc = disc_swa
    if gen_swa is not None:
        gen = gen_swa
    disc.eval()
    gen.eval()
    
    x_rec = gen(x)
    #features_real = disc.realism_analyzer.extract_features(x)
    #features_real = whiten_features(features_real, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std)
    #features_fake = disc.realism_analyzer.extract_features(x_rec)
    #features_fake = whiten_features(features_fake, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std)
    #logits_real = disc.realism_analyzer.classify_features(torch.cat((features_real, features_real), dim=1))
    #logits_fake = disc.realism_analyzer.classify_features(torch.cat((features_real, features_fake), dim=1))
    features_real = disc.realism_analyzer.extract_features(torch.cat((x, x), dim=1))
    features_real = whiten_features(features_real, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std)
    logits_real = disc.realism_analyzer.classify_features(features_real)
    features_fake = disc.realism_analyzer.extract_features(torch.cat((x, x_rec), dim=1))
    features_fake = whiten_features(features_fake, y, sub_realism_mean=sub_realism_mean, div_realism_std=div_realism_std)
    logits_fake = disc.realism_analyzer.classify_features(features_fake)
    disc_loss_realism = 0.5*hinge_loss(logits_real, 1) + 0.5*hinge_loss(logits_fake, -1)
    leakage_fake = disc.assess_leakage(x_rec)
    if train_leakage_disc_on_orig_samples:
        leakage_real = disc.assess_leakage(x)
        disc_loss_leakage = disc_leakage_loss(leakage_real, y)
    else:
        disc_loss_leakage = disc_leakage_loss(leakage_fake, y)
    disc_loss = disc_loss_realism + disc_loss_leakage
    gen_loss_realism = -logits_fake.mean()
    gen_loss_leakage = gen_leakage_loss(leakage_fake, y)
    gen_loss = gen_leakage_coefficient*gen_loss_leakage + (1-gen_leakage_coefficient)*gen_loss_realism
    
    rv = {
        'disc_loss_realism': val(disc_loss_realism),
        'disc_loss_leakage': val(disc_loss_leakage),
        'disc_loss': val(disc_loss),
        'gen_loss_realism': val(gen_loss_realism),
        'gen_loss_leakage': val(gen_loss_leakage),
        'gen_loss': val(gen_loss)
    }
    if return_example:
        rv.update({
            'orig_example': 0.5*val(to_uint8(x))+0.5,
            'rec_example': 0.5*val(to_uint8(x_rec))+0.5
        })
    return rv

def train_epoch(dataloader, *step_args, leakage_dataloader=None, return_example_idx=None, disc_steps_per_gen_step=1.0, autoencoder_gen=False, posttrain=False, **step_kwargs):
    rv = {}
    if autoencoder_gen:
        glc = step_kwargs['gen_leakage_coefficient']
        step_kwargs['gen_leakage_coefficient'] = 0.0
        step_kwargs['train_leakage_disc_on_orig_samples'] = True
        disc_steps_per_gen_step = 1.0
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx)==int else return_example_idx
    disc_steps = 0.0
    if leakage_dataloader is not None:
        dataloader_tr = zip(dataloader, leakage_dataloader)
    else:
        dataloader_tr = dataloader
    for bidx, batch in enumerate(dataloader_tr):
        if posttrain:
            step_rv = train_single_step(batch, *step_args)
        else:
            disc_steps += 1.0
            if leakage_dataloader is not None:
                batch, leakage_batch = batch
            step_rv = train_step(batch, *step_args, return_example=bidx in re_indices,
                                 leakage_batch=None if leakage_dataloader is None else leakage_batch,
                                 train_gen=disc_steps>=disc_steps_per_gen_step,
                                 **step_kwargs)
            if disc_steps >= disc_steps_per_gen_step:
                disc_steps -= disc_steps_per_gen_step
        for key, item in step_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    if 'gen_swa' in step_kwargs.keys() and step_kwargs['gen_swa'] is not None:
        torch.optim.swa_utils.update_bn(dataloader, step_kwargs['gen_swa'])
    if 'disc_swa' in step_kwargs.keys() and step_kwargs['disc_swa'] is not None:
        torch.optim.swa_utils.update_bn(dataloader, step_kwargs['disc_swa'])
    for key, item in rv.items():
        if not key in ['orig_example', 'rec_example']:
            rv[key] = np.nanmean(item)
    if autoencoder_gen:
        step_kwargs['gen_leakage_coefficient'] = glc
        step_kwargs['train_leakage_disc_on_orig_samples'] = False
    return rv

def eval_epoch(dataloader, *step_args, return_example_idx=None, autoencoder_gen=False, posttrain=False, **step_kwargs):
    rv = {}
    if autoencoder_gen:
        glc = step_kwargs['gen_leakage_coefficient']
        step_kwargs['gen_leakage_coefficient'] = 0.0
        step_kwargs['train_leakage_disc_on_orig_samples'] = True
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx)==int else return_example_idx
    for bidx, batch in enumerate(dataloader):
        if posttrain:
            step_rv = eval_single_step(batch, *step_args)
        else:
            step_rv = eval_step(batch, *step_args, return_example=bidx in re_indices, **step_kwargs)
        for key, item in step_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    for key, item in rv.items():
        if not key in ['orig_example', 'rec_example']:
            rv[key] = np.nanmean(item)
    if autoencoder_gen:
        step_kwargs['gen_leakage_coefficient'] = glc
        step_kwargs['train_leakage_disc_on_orig_samples'] = False
    return rv