import numpy as np
import torch
from torch import nn, optim

def val(tensor):
    try:
        return tensor.detach().cpu().numpy()
    except:
        return np.nan

def acc(logits, y):
    logits, y = val(logits), val(y)
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(np.equal(predictions, y))
    return acc

def hinge_acc(logits, y):
    logits = val(logits)
    predictions = np.sign(logits)
    acc = np.mean(np.equal(predictions, y*np.ones_like(predictions)))
    return acc

def to_uint8(x):
    x = 255.0*(0.5*x+0.5)
    x = x.to(torch.uint8).to(torch.float)
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

def kl_div(logits_p, logits_q):
    p = nn.functional.softmax(logits_p, dim=-1)
    logp = nn.functional.log_softmax(logits_p, dim=-1)
    logq = nn.functional.log_softmax(logits_q, dim=-1)
    out = (p*(logp-logq)).sum(dim=-1).mean()
    return out

def gradient_penalty(real_data, fake_data, disc, device):
    alpha = torch.rand(real_data.size(0), 1, 1, 1, device=device)
    interpolated_data = alpha*real_data + (1-alpha)*fake_data
    interpolated_data.requires_grad_(True)
    interpolated_features = disc.extract_features(interpolated_data)
    realism_logits = disc.classify_realism(interpolated_features)
    grads = torch.autograd.grad(outputs=realism_logits, inputs=interpolated_data,
                                grad_outputs=torch.ones(realism_logits.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = (grads.norm(2, dim=1)**2).mean()
    return gradient_penalty

@torch.no_grad()
def calculate_mean_accuracy(dataloader, gen, disc, device, downstream=False, y_clamp=None):
    gen.eval()
    disc.eval()
    if downstream:
        acc_downstream = 0.0
    else:
        acc_orig_labels = 0.0
        acc_rec_labels = 0.0
    for bidx, batch in enumerate(dataloader):
        x, y, mdata = batch
        y_orig = mdata['target']
        x, y, y_orig = x.to(device), y.to(device), y_orig.to(device)
        if y_clamp is None:
            y_clamp_ = torch.randint(0, gen.num_leakage_classes, dtype=y.dtype, device=y.device, size=y.size())
        else:
            y_clamp_ = y_clamp*torch.ones_like(y)
        x_rec = to_uint8(gen(x, y_clamp_))
        logits_rec = disc(x_rec)
        if downstream:
            acc_downstream += acc(logits_rec, y_orig)
        else:
            acc_orig_labels += acc(logits_rec, y)
            acc_rec_labels += acc(logits_rec, y_clamp_)
    if downstream:
        acc_downstream /= len(dataloader)
        return acc_downstream
    else:
        acc_orig_labels /= len(dataloader)
        acc_rec_labels /= len(dataloader)
        return acc_orig_labels, acc_rec_labels

@torch.no_grad()
def calculate_class_conditional_inception_score(dataloader, gen, disc, device, y_clamp=0):
    gen.eval()
    disc.eval()
    mean_logits_orig, mean_logits_rec = None, None
    for bidx, batch in enumerate(dataloader):
        x, y, _ = batch
        x, y = x.to(device), y.to(device)
        if y_clamp is None:
            y_clamp_ = torch.randint(0, gen.num_leakage_classes, dtype=y.dtype, device=y.device, size=y.size())
        else:
            y_clamp_ = y_clamp*torch.ones_like(y)
        x_rec = to_uint8(gen(x, y_clamp_))
        if len(x[y==y_clamp]) > 0:
            logits_orig = disc(x[y==y_clamp]).mean(dim=0)
            if mean_logits_orig is None:
                mean_logits_orig = logits_orig
            else:
                mean_logits_orig = (1/(bidx+1))*logits_orig + (bidx/(bidx+1))*mean_logits_orig
        logits_rec = disc(x_rec).mean(dim=0)
        if mean_logits_rec is None:
            mean_logits_rec = logits_rec
        else:
            mean_logits_rec = (1/(bidx+1))*logits_rec + (bidx/(bidx+1))*mean_logits_rec
    inception_score = kl_div(mean_logits_orig, mean_logits_rec)
    return val(inception_score)

@torch.no_grad()
def calculate_inception_score(dataloader, gen, disc, device):
    gen.eval()
    disc.eval()
    inception_score = 0.0
    for batch in dataloader:
        x, _, _ = batch
        y_clamp = torch.randint(0, gen.num_leakage_classes, device=x.device, dtype=torch.long, size=(x.size(0),))
        x = x.to(device)
        x_rec = to_uint8(gen(x, y_clamp))
        logits_orig = disc(x)
        logits_rec = disc(x_rec)
        batch_is = kl_div(logits_orig, logits_rec)
        inception_score += batch_is
    inception_score = inception_score/len(dataloader)
    return val(inception_score)

def apply_mixup_to_data(x, y, alpha):
    lbd = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    x = lbd*x + (1-lbd)*x[index, :]
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lbd

def apply_mixup_to_criterion(criterion, logits, y_a, y_b, lbd):
    return lbd*criterion(logits, y_a) + (1-lbd)*criterion(logits, y_b)

def train_single_step(batch, model, opt, loss_fn, device, original_target=False):
    if original_target:
        x, _, mdata = batch
        y = mdata['target']
    else:
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
def eval_single_step(batch, model, loss_fn, device, original_target=False):
    if original_target:
        x, _, mdata = batch
        y = mdata['target']
    else:
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

def train_step(batch, gen, gen_opt, disc, disc_opt, device, y_clamp=0, l1_rec_coefficient=0.0, gen_leakage_coefficient=0.5,
               return_example=False, return_weight_norms=False, return_grad_norms=False, pretrain=False, disc_grad_penalty=0.0,
               mixup_alpha=0.0, average_deviation_penalty=0.0, train_gen=True, train_disc=True, **kwargs):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    if y_clamp is None:
        y_clamp = torch.randint(0, gen.num_leakage_classes, size=y.size(), dtype=y.dtype, device=y.device)
    else:
        y_clamp *= torch.ones_like(y)
    disc.train()
    gen.train()
    rv = {}
    
    if train_disc:
        with torch.no_grad():
            x_rec = gen(x, y_clamp)
        x_leakage, x_rec_leakage = x[:len(x)//2], x_rec[:len(x_rec)//2]
        x_realism, x_rec_realism = x[len(x)//2:], x_rec[len(x_rec)//2:]
        if mixup_alpha > 0:
            x_leakage, y_orig_leakage_a, y_orig_leakage_b, lbd_orig = apply_mixup_to_data(x_leakage, y[:len(x)//2], mixup_alpha)
            x_rec_leakage, y_rec_leakage_a, y_rec_leakage_b, lbd_rec = apply_mixup_to_data(x_rec_leakage, y[:len(x)//2], mixup_alpha)
            
        disc_features = disc.extract_features(torch.cat((x_leakage, x_realism, x_rec_leakage, x_rec_realism), dim=0))
        disc_features_orig_leakage = disc_features[:len(x)//2]
        disc_features_orig_realism = disc_features[len(x)//2:len(x)]
        disc_features_rec_leakage = disc_features[len(x):len(x)+len(x)//2]
        disc_features_rec_realism = disc_features[len(x)+len(x)//2:]
        
        disc_logits_leakage = disc.classify_leakage(torch.cat((disc_features_orig_leakage, disc_features_rec_leakage), dim=0))
        disc_logits_orig_leakage = disc_logits_leakage[:len(x)//2]
        disc_logits_rec_leakage = disc_logits_leakage[len(x)//2:]
        if mixup_alpha > 0:
            disc_leakage_loss_orig = apply_mixup_to_criterion(
                nn.functional.multi_margin_loss, disc_logits_orig_leakage, y_orig_leakage_a, y_orig_leakage_b, lbd_orig
            )
            disc_leakage_loss_rec = apply_mixup_to_criterion(
                nn.functional.multi_margin_loss, disc_logits_rec_leakage, y_rec_leakage_a, y_rec_leakage_b, lbd_rec
            )
        else:
            disc_leakage_loss_orig = nn.functional.multi_margin_loss(disc_logits_orig_leakage, y[:len(x)//2])
            disc_leakage_loss_rec = nn.functional.multi_margin_loss(disc_logits_rec_leakage, y[:len(x)//2])
        disc_leakage_loss = 0.5*disc_leakage_loss_orig + 0.5*disc_leakage_loss_rec
        
        disc_logits_realism = disc.classify_realism(torch.cat((disc_features_orig_realism, disc_features_rec_realism), dim=0))
        disc_logits_orig_realism = disc_logits_realism[:len(x)//2]
        disc_logits_rec_realism = disc_logits_realism[len(x)//2:]
        disc_realism_loss_orig = hinge_loss(disc_logits_orig_realism, 1)
        disc_realism_loss_rec = hinge_loss(disc_logits_rec_realism, -1)
        disc_realism_loss = 0.5*disc_realism_loss_orig + 0.5*disc_realism_loss_rec
        
        disc_loss = 0.5*disc_leakage_loss + 0.5*disc_realism_loss
        if disc_grad_penalty != 0.0:
            disc_gp = gradient_penalty(x, x_rec, disc, device)
            disc_loss = disc_loss + disc_grad_penalty*disc_gp
        disc_opt.zero_grad(set_to_none=True)
        disc_loss.backward()
        if return_grad_norms:
            rv.update({'disc_grad_norm': get_grad_norm(disc)})
        disc_opt.step()
    else:
        disc_loss = disc_leakage_loss = disc_realism_loss = None
    
    if train_gen:
        x_rec = gen(x, y_clamp)
        disc_features_rec = disc.extract_features(x_rec)
        
        disc_leakage_predictions_rec = disc.classify_leakage(disc_features_rec)
        gen_leakage_loss_neg = torch.gather(disc_leakage_predictions_rec, 1, y_clamp.unsqueeze(1)).mean()
        gen_leakage_loss_pos = torch.gather(
            disc_leakage_predictions_rec, 1,
            torch.tensor([[j for j in range(disc_leakage_predictions_rec.size(1)) if j!=yy] for yy in y_clamp],
                         dtype=torch.long, device=y_clamp.device)).mean()
        gen_leakage_loss = gen_leakage_loss_pos - gen_leakage_loss_neg
        
        disc_realism_predictions_rec = disc.classify_realism(disc_features_rec)
        gen_realism_loss = -disc_realism_predictions_rec.mean()
        gen_l1_reconstruction_loss = nn.functional.l1_loss(x_rec, x)
        
        if not pretrain:
            gen_loss = gen_leakage_coefficient*gen_leakage_loss + \
                       (1-gen_leakage_coefficient)*gen_realism_loss + \
                       l1_rec_coefficient*gen_l1_reconstruction_loss
            if average_deviation_penalty != 0.0:
                gen_loss = gen_loss + average_deviation_penalty*gen.get_avg_departure_penalty()
        else:
            gen_loss = gen_l1_reconstruction_loss
        gen_opt.zero_grad(set_to_none=True)
        gen_loss.backward()
        if return_grad_norms:
            rv.update({'gen_grad_norm': get_grad_norm(gen)})
        gen_opt.step()
        if not pretrain and average_deviation_penalty != 0.0:
            gen.update_avg()
        elif average_deviation_penalty != 0.0:
            gen.reset_avg()
    else:
        gen_loss = gen_leakage_loss = gen_realism_loss = gen_l1_reconstruction_loss = None
    
    rv.update({
        'disc_realism_loss': val(disc_realism_loss),
        'disc_leakage_loss': val(disc_leakage_loss),
        'disc_loss': val(disc_loss),
        'disc_realism_acc_orig': hinge_acc(disc_logits_orig_realism, 1),
        'disc_realism_acc_rec': hinge_acc(disc_logits_rec_realism, -1),
        'disc_realism_acc': 0.5*hinge_acc(disc_logits_orig_realism, 1) + 0.5*hinge_acc(disc_logits_rec_realism, -1),
        'disc_leakage_acc_orig': acc(disc_logits_orig_leakage, y[:len(x)//2]),
        'disc_leakage_acc_rec': acc(disc_logits_rec_leakage, y[:len(x)//2]),
        'disc_leakage_acc': 0.5*acc(disc_logits_orig_leakage, y[:len(x)//2]) + 0.5*acc(disc_logits_rec_leakage, y[:len(x)//2]),
        'gen_realism_loss': val(gen_realism_loss),
        'gen_leakage_loss': val(gen_leakage_loss),
        'gen_l1_reconstruction_loss': val(gen_l1_reconstruction_loss),
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
def eval_step(batch, gen, disc, device, y_clamp=0, l1_rec_coefficient=0.0,
              gen_leakage_coefficient=0.5, return_example=False, **kwargs):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    if y_clamp is None:
        y_clamp = torch.randint(0, gen.num_leakage_classes, dtype=y.dtype, device=y.device, size=y.size())
    else:
        y_clamp *= torch.ones_like(y)
    disc.eval()
    gen.eval()
    rv = {}
    
    x_rec = gen(x, y_clamp)
    disc_features_orig = disc.extract_features(x)
    disc_features_rec = disc.extract_features(x_rec)
    
    disc_leakage_predictions_orig = disc.classify_leakage(disc_features_orig)
    disc_leakage_predictions_rec = disc.classify_leakage(disc_features_rec)
    disc_leakage_loss_orig = nn.functional.multi_margin_loss(disc_leakage_predictions_orig, y)
    disc_leakage_loss_rec = nn.functional.multi_margin_loss(disc_leakage_predictions_rec, y)
    disc_leakage_loss = 0.5*disc_leakage_loss_orig + 0.5*disc_leakage_loss_rec
    gen_leakage_loss_neg = torch.gather(disc_leakage_predictions_rec, 1, y_clamp.unsqueeze(1)).mean()
    gen_leakage_loss_pos = torch.gather(
        disc_leakage_predictions_rec, 1,
        torch.tensor([[j for j in range(disc_leakage_predictions_rec.size(1)) if j!=yy] for yy in y_clamp],
                     device=y_clamp.device, dtype=torch.long)).mean()
    gen_leakage_loss = gen_leakage_loss_pos - gen_leakage_loss_neg
    
    disc_realism_predictions_orig = disc.classify_realism(disc_features_orig)
    disc_realism_predictions_rec = disc.classify_realism(disc_features_rec)
    disc_realism_loss_orig = hinge_loss(disc_realism_predictions_orig, 1)
    disc_realism_loss_rec = hinge_loss(disc_realism_predictions_rec, -1)
    disc_realism_loss = 0.5*disc_realism_loss_orig + 0.5*disc_realism_loss_rec
    gen_realism_loss = -disc_realism_predictions_rec.mean()
    gen_l1_reconstruction_loss = nn.functional.l1_loss(x_rec, x)
    
    disc_loss = 0.5*disc_leakage_loss + 0.5*disc_realism_loss
    gen_loss = gen_leakage_coefficient*gen_leakage_loss + \
               (1-gen_leakage_coefficient)*gen_realism_loss + \
               l1_rec_coefficient*gen_l1_reconstruction_loss
    
    rv.update({
        'disc_leakage_loss': val(disc_leakage_loss),
        'disc_realism_loss': val(disc_realism_loss),
        'disc_loss': val(disc_loss),
        'disc_realism_acc_orig': hinge_acc(disc_realism_predictions_orig, 1),
        'disc_realism_acc_rec': hinge_acc(disc_realism_predictions_rec, -1),
        'disc_realism_acc': 0.5*hinge_acc(disc_realism_predictions_orig, 1) + 0.5*hinge_acc(disc_realism_predictions_rec, -1),
        'disc_leakage_acc_orig': acc(disc_leakage_predictions_orig, y),
        'disc_leakage_acc_rec': acc(disc_leakage_predictions_rec, y),
        'disc_leakage_acc': 0.5*acc(disc_leakage_predictions_orig, y) + 0.5*acc(disc_leakage_predictions_rec, y),
        'gen_leakage_loss': val(gen_leakage_loss),
        'gen_realism_loss': val(gen_realism_loss),
        'gen_l1_reconstruction_loss': val(gen_l1_reconstruction_loss),
        'gen_loss': val(gen_loss)
    })
    if return_example:
        examples_per_class = 100//gen.num_leakage_classes
        orig_examples = x[:gen.num_leakage_classes*examples_per_class]
        y_clamp = torch.cat([leakage_class*torch.ones_like(y[:examples_per_class])
                             for leakage_class in range(gen.num_leakage_classes)], dim=0)
        rec_examples = gen(orig_examples, y_clamp)
        rv.update({
            'orig_example': 0.5*val(to_uint8(orig_examples))+0.5,
            'rec_example': 0.5*val(to_uint8(rec_examples))+0.5
        })
    return rv

def train_epoch(dataloader, *step_args, return_example_idx=None, disc_steps_per_gen_step=1.0, pretrain=False, posttrain=False, **step_kwargs):
    rv = {}
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx)==int else return_example_idx
    disc_steps = gen_steps = 0.0
    def run_epoch():
        nonlocal rv, disc_steps, gen_steps
        for bidx, batch in enumerate(dataloader):
            if posttrain:
                step_rv = train_single_step(batch, *step_args, original_target=step_kwargs['original_target'])
            else:
                if disc_steps_per_gen_step > 1:
                    disc_steps += 1
                    train_disc = True
                    train_gen = disc_steps >= disc_steps_per_gen_step
                    if disc_steps >= disc_steps_per_gen_step:
                        disc_steps -= disc_steps_per_gen_step
                elif 1/disc_steps_per_gen_step > 1:
                    gen_steps += 1
                    train_gen = True
                    train_disc = gen_steps >= 1/disc_steps_per_gen_step
                    if gen_steps >= 1/disc_steps_per_gen_step:
                        gen_steps -= 1/disc_steps_per_gen_step
                else:
                    train_disc = train_gen = True
                step_rv = train_step(
                    batch, *step_args,
                    return_example=bidx in re_indices, train_gen=train_gen, train_disc=train_disc, pretrain=pretrain, **step_kwargs
                )
            for key, item in step_rv.items():
                if not key in rv.keys():
                    rv[key] = []
                rv[key].append(item)
    run_epoch()
    for key, item in rv.items():
        if not key in ['orig_example', 'rec_example']:
            rv[key] = np.nanmean(item)
    return rv

def eval_epoch(dataloader, *step_args, return_example_idx=None, posttrain=False, 
               leakage_eval_disc=None, downstream_eval_disc=None, **step_kwargs):
    rv = {}
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx)==int else return_example_idx
    for bidx, batch in enumerate(dataloader):
        if posttrain:
            step_rv = eval_single_step(batch, *step_args, original_target=step_kwargs['original_target'])
        else:
            step_rv = eval_step(batch, *step_args, return_example=bidx in re_indices, **step_kwargs)
        for key, item in step_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    if leakage_eval_disc is not None:
        gen = step_args[0]
        device = step_args[-1]
        y_clamp = step_kwargs['y_clamp']
        acc_orig_labels, acc_rec_labels = calculate_mean_accuracy(dataloader, gen, leakage_eval_disc, device, y_clamp=y_clamp)
        rv['acc_leakage_orig_labels'] = acc_orig_labels
        rv['acc_leakage_rec_labels'] = acc_rec_labels
    if downstream_eval_disc is not None:
        gen = step_args[0]
        device = step_args[-1]
        acc_downstream = calculate_mean_accuracy(dataloader, gen, downstream_eval_disc, device, downstream=True, y_clamp=y_clamp)
        rv['acc_downstream'] = acc_downstream
    for key, item in rv.items():
        if not key in ['orig_example', 'rec_example']:
            rv[key] = np.nanmean(item)
    return rv
    