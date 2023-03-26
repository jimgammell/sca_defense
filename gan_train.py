import numpy as np
import torch
from torch import nn, optim

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

def hinge_realism_loss(logits):
    return -logits.mean()
    
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
    #return (((x - x.mean(dim=0, keepdim=True))**2).mean(dim=0, keepdim=True))**0.5
    return x.std(dim=0, keepdim=True)

# Might suffer from class imbalance.
def get_mcmatching_penalty(X, y):
    def get_mn(x):
        return x.mean(dim=0, keepdim=True)
    def get_cov(x, x_mn):
        return torch.mm(x.permute(1, 0), x)/x.size(0) - torch.mm(x_mn.permute(1, 0), x)
    X_mn = get_mn(X)
    X_cov = get_cov(X, X_mn)
    loss = torch.tensor(0.0, dtype=torch.float, device=X.device, requires_grad=True)
    for yy in torch.unique(y):
        cc_mn = get_mn(X[y==yy])
        cc_cov = get_cov(X[y==yy], cc_mn)
        loss += (X_mn-cc_mn).norm(p=1)
        loss += (X_cov-cc_cov).norm(p=1)
    loss /= len(X)
    return loss

def get_whitened_features(features, y, detach=False):
    mn = torch.tensor(0.0, requires_grad=not(detach), device=features.device)
    std = torch.tensor(1.0, requires_grad=not(detach), device=features.device)
    for yy in torch.unique(y):
        mn_yy = features[y==yy].mean(dim=0, keepdim=True)
        if detach:
            mn_yy = mn_yy.detach()
        features[y==yy] = features[y==yy] - mn_yy
        std_yy = get_std(features[y==yy])
        if detach:
            std_yy = std_yy.detach()
        features[y==yy] = features[y==yy] / std_yy
        mn = mn + mn_yy
        std = std * std_yy
    mn = mn / len(torch.unique(y))
    std = std ** (1.0/len(torch.unique(y)))
    if detach:
        mn, std = mn.detach(), std.detach()
    features = std * features + mn
    return features

def feature_whitening_penalty(features, y):
    mean = torch.tensor(0.0, requires_grad=True, device=features.device)
    std_dev = torch.tensor(1.0, requires_grad=True, device=features.device)
    unique_labels = torch.unique(y)
    cc_means, cc_std_devs = [], []
    for label in unique_labels:
        cc_mean = features[y==label].mean(dim=0, keepdim=True)
        cc_std_dev = get_std(features[y==label])
        cc_means.append(cc_mean)
        cc_std_devs.append(cc_std_dev)
        mean = mean + cc_mean
        std_dev = std_dev * cc_std_dev
    mean = mean / len(unique_labels)
    std_dev = std_dev ** (1.0/len(unique_labels))
    loss = torch.tensor(0.0, requires_grad=True, device=features.device)
    for cc_mean in cc_means:
        loss = loss + (mean - cc_mean).norm(p=1)
    for cc_std_dev in cc_std_devs:
        loss = loss + (std_dev - cc_std_dev).norm(p=1)
    loss = loss / len(features)
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
               whiten_features=False,
               gen_leakage_coefficient=0.5, disc_leakage_coefficient=0.5, disc_invariance_coefficient=1.0,
               train_gen=True, train_disc=True, 
               disc_realism_loss_fn=hinge_loss, gen_realism_loss_fn=hinge_realism_loss,
               disc_leakage_loss_fn=nn.functional.multi_margin_loss, gen_leakage_loss_fn=stats_l1_loss,
               train_leakage_disc_on_orig_samples=False, clip_gradients=False,
               whitening_averaging_coefficient=0.0,
               detached_feature_whitening=False,
               gen_swa=None, disc_swa=None):
    
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    disc.train()
    gen.train()
    gen_tr = gen_swa if gen_swa is not None else gen
    disc_tr = disc_swa if disc_swa is not None else disc
    gen_tr.train()
    disc_tr.train()
    rv = {}
    
    if train_disc:
        # feature extraction for subsequent updates
        if leakage_batch is not None:
            x_lk, y_lk, _ = leakage_batch
            x_lk, y_lk = x_lk.to(device), y_lk.to(device)
        else:
            x_lk, y_lk = x, y
        with torch.no_grad():
            x_rec_lk = gen_tr(x_lk)
        features_orig = disc.extract_features(x_lk)
        features_rec = disc.extract_features(x_rec_lk)
        
        # calculate loss w.r.t. distinguishing real vs fake samples
        features_realism_orig = disc.get_realism_features(features_orig)
        features_realism_rec = disc.get_realism_features(features_rec)
        invariance_penalty = feature_whitening_penalty(features_realism_orig, y_lk)
        features_realism_ref = features_realism_orig.clone()
        if whiten_features:
            features_realism_orig = get_whitened_features(features_realism_orig, y_lk, detach=detached_feature_whitening)
        logits_realism_orig = disc.classify_realism(features_realism_ref, features_realism_orig)
        logits_realism_rec = disc.classify_realism(features_realism_ref, features_realism_rec)
        disc_loss_realism = \
            0.5*disc_realism_loss_fn(logits_realism_orig, 1) +\
            0.5*disc_realism_loss_fn(logits_realism_rec, -1) +\
            disc_invariance_coefficient*invariance_penalty
        
        # calculate loss w.r.t. identifying leaked information from samples
        if train_leakage_disc_on_orig_samples:
            leakage_features = disc.get_leakage_features(features_orig)
        else:
            leakage_features = disc.get_leakage_features(features_rec)
        leakage_logits = disc.classify_leakage(leakage_features)
        disc_loss_leakage = disc_leakage_loss_fn(leakage_logits, y_lk)
        
        # update discriminator parameters
        disc_loss = (1-disc_leakage_coefficient)*disc_loss_realism + disc_leakage_coefficient*disc_loss_leakage
        disc_opt.zero_grad(set_to_none=True)
        disc_loss.backward()
        if clip_gradients:
            nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
        if return_grad_norms:
            rv.update({'disc_grad_norm': get_grad_norm(disc)})
        disc_opt.step()
        if disc_swa is not None:
            disc_swa.update_parameters(disc)
    else:
        disc_loss = disc_loss_realism = disc_loss_leakage = invariance_penalty = None
    
    # generator update
    if train_gen:
        # feature extraction for subsequent batches
        x_rec = gen(x)
        with torch.no_grad():
            features_orig = disc_tr.extract_features(x)
            features_realism_ref = disc_tr.get_realism_features(features_orig)
        features_rec = disc_tr.extract_features(x_rec)
        features_realism_rec = disc_tr.get_realism_features(features_rec)
        features_leakage_rec = disc_tr.get_leakage_features(features_rec)
        
        # calculate loss w.r.t. making fake images similar to real images
        logits_realism = disc_tr.classify_realism(features_realism_ref, features_realism_rec)
        gen_loss_realism = gen_realism_loss_fn(logits_realism)
        
        # calculate loss w.r.t. avoiding information leakage
        logits_leakage = disc_tr.classify_leakage(features_leakage_rec)
        gen_loss_leakage = gen_leakage_loss_fn(logits_leakage, y)
        
        # update generator parameters
        gen_loss = gen_leakage_coefficient*gen_loss_leakage + (1-gen_leakage_coefficient)*gen_loss_realism
        if project_gen_updates: # and leakage loss > threshold
            dot = lambda x, y: (x*y).sum()
            gen_opt.zero_grad(set_to_none=True)
            gen_loss_leakage.backward(retain_graph=True)
            gen_leakage_gradients = [param.grad.clone() for param in gen.parameters()]
            gen_opt.zero_grad(set_to_none=True)
            gen_loss_realism.backward()
            for leakage_grad, param in zip(gen_leakage_gradients, gen.parameters()):
                realism_grad = param.grad
                projected_grad = realism_grad - leakage_grad*dot(leakage_grad, realism_grad)/(dot(leakage_grad, leakage_grad)+1e-12)
                projected_grad = 0.5*projected_grad + 0.5*leakage_grad
                param.grad = projected_grad
        else:
            gen_opt.zero_grad(set_to_none=True)
            gen_loss.backward()
        if clip_gradients:
            nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        if return_grad_norms:
            rv.update({'gen_grad_norm': get_grad_norm(gen)})
        gen_opt.step()
        if gen_swa is not None:
            gen_swa.update_parameters(gen)
    else:
        gen_loss = gen_loss_realism = gen_loss_leakage = None
    
    rv.update({
        'disc_loss_realism': val(disc_loss_realism),
        'disc_loss_leakage': val(disc_loss_leakage),
        'disc_loss': val(disc_loss),
        'disc_invariance_penalty': val(invariance_penalty),
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

# To try: use whitened features only as the reference
@torch.no_grad()
def eval_step(batch, gen, disc, device,
              return_example=False, whiten_features=False,
              gen_leakage_coefficient=0.5, disc_leakage_coefficient=0.5, disc_invariance_coefficient=1.0,
              train_leakage_disc_on_orig_samples=False,
              whitening_averaging_coefficient=0.0,
              disc_swa=None, gen_swa=None,
              detached_feature_whitening=False,
              disc_realism_loss_fn=hinge_loss, gen_realism_loss_fn=hinge_realism_loss,
              disc_leakage_loss_fn=nn.functional.multi_margin_loss, gen_leakage_loss_fn=stats_l1_loss, **kwargs):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    if disc_swa is not None:
        disc = disc_swa
    if gen_swa is not None:
        gen = gen_swa
    disc.eval()
    gen.eval()
    
    x_rec = gen(x)
    features_orig = disc.extract_features(x)
    features_rec = disc.extract_features(x_rec)
    features_realism_orig = disc.get_realism_features(features_orig)
    features_realism_rec = disc.get_realism_features(features_rec)
    features_realism_ref = features_realism_orig.clone()
    features_leakage_orig = disc.get_leakage_features(features_orig)
    features_leakage_rec = disc.get_leakage_features(features_rec)
    invariance_penalty = feature_whitening_penalty(features_realism_orig, y)
    if whiten_features:
        features_realism_orig = get_whitened_features(features_realism_orig, y)
    logits_realism_orig = disc.classify_realism(features_realism_ref, features_realism_orig)
    logits_realism_rec = disc.classify_realism(features_realism_ref, features_realism_rec)
    disc_loss_realism = \
        0.5*disc_realism_loss_fn(logits_realism_orig, 1) +\
        0.5*disc_realism_loss_fn(logits_realism_rec, -1) +\
        disc_invariance_coefficient*invariance_penalty
    logits_leakage_rec = disc.classify_leakage(features_leakage_rec)
    if train_leakage_disc_on_orig_samples:
        logits_leakage_orig = disc.classify_leakage(features_leakage_orig)
        disc_loss_leakage = disc_leakage_loss_fn(logits_leakage_orig, y)
    else:
        disc_loss_leakage = disc_leakage_loss_fn(logits_leakage_rec, y)
    disc_loss = (1-disc_leakage_coefficient)*disc_loss_realism + disc_leakage_coefficient*disc_loss_leakage
    gen_loss_realism = gen_realism_loss_fn(logits_realism_rec)
    gen_loss_leakage = gen_leakage_loss_fn(logits_leakage_rec, y)
    gen_loss = gen_leakage_coefficient*gen_loss_leakage + (1-gen_leakage_coefficient)*gen_loss_realism
    
    rv = {
        'disc_loss_realism': val(disc_loss_realism),
        'disc_loss_leakage': val(disc_loss_leakage),
        'disc_loss': val(disc_loss),
        'disc_invariance_penalty': val(invariance_penalty),
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

def train_epoch(dataloader, *step_args, leakage_dataloader=None, return_example_idx=None, disc_steps_per_gen_step=1.0, autoencoder_gen=False, posttrain=False, profile_epoch=False, **step_kwargs):
    rv = {}
    if autoencoder_gen:
        glc = step_kwargs['gen_leakage_coefficient']
        step_kwargs['gen_leakage_coefficient'] = 0.0
        step_kwargs['train_leakage_disc_on_orig_samples'] = True
        disc_steps_per_gen_step = 1.0
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx)==int else return_example_idx
    disc_steps = gen_steps = 0.0
    if leakage_dataloader is not None:
        dataloader_tr = zip(dataloader, leakage_dataloader)
    else:
        dataloader_tr = dataloader
    def run_epoch():
        nonlocal rv, disc_steps, gen_steps
        for bidx, batch in enumerate(dataloader_tr):
            if posttrain:
                step_rv = train_single_step(batch, *step_args)
            else:
                if disc_steps_per_gen_step > 1:
                    disc_steps += 1
                    train_disc = True
                    train_gen = disc_steps >= disc_steps_per_gen_step
                    if disc_steps >= disc_steps_per_gen_step:
                        disc_steps -= disc_steps_per_gen_step
                elif disc_steps_per_gen_step < 1:
                    gen_steps += 1
                    train_gen = True
                    train_disc = gen_steps >= 1/disc_steps_per_gen_step
                    if gen_steps >= 1/disc_steps_per_gen_step:
                        gen_steps -= 1/disc_steps_per_gen_step
                else:
                    train_disc = train_gen = True
                if leakage_dataloader is not None:
                    batch, leakage_batch = batch
                step_rv = train_step(batch, *step_args, return_example=bidx in re_indices,
                                     leakage_batch=None if leakage_dataloader is None else leakage_batch,
                                     train_gen=train_gen, train_disc=train_disc,
                                     **step_kwargs)
                if profile_epoch:
                    prof.step()
            for key, item in step_rv.items():
                if not key in rv.keys():
                    rv[key] = []
                rv[key].append(item)
    if profile_epoch:
        with  torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs/saunet_gan'),
            record_shapes=True,
            with_stack=True) as prof:
            run_epoch()
    else:
        run_epoch()
    if 'gen_swa' in step_kwargs.keys() and step_kwargs['gen_swa'] is not None:
        torch.optim.swa_utils.update_bn(dataloader, step_kwargs['gen_swa'])
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