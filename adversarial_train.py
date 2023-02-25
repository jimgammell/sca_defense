import numpy as np
import torch
from torch import nn, optim

class ModelConfusion(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, logits):
        return nn.functional.binary_cross_entropy(
            nn.functional.softmax(logits, dim=-1),
            nn.functional.softmax(torch.zeros_like(logits), dim=-1))

def compute_confusing_example(example, disc, gen,
                              eps=1e-5,
                              warmup_iter=100,
                              max_iter=10000,
                              opt_const=optim.Adam,
                              opt_kwargs={'lr': 1e-2}):
    adv_example = gen(example).detach()
    adv_example.requires_grad = True
    opt = opt_const([adv_example], **opt_kwargs)
    criterion = ModelConfusion()
    orig_loss = np.inf
    for i in range(max_iter):
        logits = disc(nn.functional.hardtanh(adv_example))
        loss = criterion(logits)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i>=warmup_iter and np.abs(orig_loss-loss.cpu().detach().numpy()) < eps:
            break
        orig_loss = loss.detach().cpu().numpy()
    return nn.functional.hardtanh(adv_example.detach()), loss.detach().cpu().numpy()

def to_uint8(x):
    x = 255.0*(0.5*x+0.5) # range of [-1, 1) to [0, 256)
    x = x.to(torch.uint8) # quantize to {0, ..., 255}
    x = x.to(torch.float) # convert quantized tensor back to float for compatibility w/ nns
    x = 2.0*(x/255.0)-1.0 # back to range of [-1, 1)
    return x

def pretrain_gen_step(batch, gen, gen_opt, gen_loss_fn, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    
    gen.train()
    gen_logits = gen(x)
    gen_loss = gen_loss_fn(gen_logits, x)
    gen_opt.zero_grad()
    gen_loss.backward()
    gen_opt.step()
    
    return {
        'gen_loss': gen_loss.detach().cpu().numpy()
    }

def eval_gen_step(batch, gen, gen_loss_fn, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    
    gen.eval()
    gen_logits = gen(x)
    gen_loss = gen_loss_fn(gen_logits, x)
    
    return {
        'gen_loss': gen_loss.detach().cpu().numpy()
    }

def pretrain_disc_step(batch, disc, disc_opt, disc_loss_fn, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    
    disc.train()
    disc_logits = disc(x)
    disc_loss = disc_loss_fn(disc_logits, y)
    disc_opt.zero_grad()
    disc_loss.backward()
    disc_opt.step()
    disc_acc = np.mean(
        np.equal(
            np.argmax(disc_logits.detach().cpu().numpy(), axis=-1),
            y.detach().cpu().numpy()
        )
    )
    
    return {
        'disc_loss': disc_loss.detach().cpu().numpy(),
        'disc_acc': disc_acc
    }

def eval_disc_step(batch, disc, disc_loss_fn, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    
    disc.eval()
    disc_logits = disc(x)
    disc_loss = disc_loss_fn(disc_logits, y)
    disc_acc = np.mean(
        np.equal(
            np.argmax(disc_logits.detach().cpu().numpy(), axis=-1),
            y.detach().cpu().numpy()
        )
    )
    
    return {
        'disc_loss': disc_loss.detach().cpu().numpy(),
        'disc_acc': disc_acc
    }

def train_evaldisc_step(batch, gen, eval_disc, eval_disc_opt, eval_disc_loss_fn, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    gen.eval()
    eval_disc.train()
    confusing_example = gen(x)
    confusing_example = to_uint8(confusing_example)
    eval_disc_logits = eval_disc(confusing_example)
    eval_disc_loss = eval_disc_loss_fn(eval_disc_logits, y)
    eval_disc_opt.zero_grad()
    eval_disc_loss.backward()
    eval_disc_opt.step()
    eval_disc_acc = np.mean(
        np.equal(
            np.argmax(eval_disc_logits.detach().cpu().numpy(), axis=-1),
            y.detach().cpu().numpy()
        )
    )
    
    return {
        'eval_disc_loss': eval_disc_loss.detach().cpu().numpy(),
        'eval_disc_acc': eval_disc_acc
    }

def train_step(batch, disc, gen, disc_opt, gen_opt, disc_loss_fn, gen_loss_fn, device,
               return_example=False,
               project_rec_updates=True,
               disc_orig_sample_prob=0.0,
               ce_kwargs={},
               loss_mixture_coefficient=0.5,
               reconstruction_critic_items=None):
    if return_example:
        raise NotImplementedError
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    disc.eval()
    confusing_example, confusing_example_loss = compute_confusing_example(x, disc, gen)
    
    disc.train()
    if disc_orig_sample_prob > np.random.uniform(0, 1):
        disc_logits = disc(x)
    else:
        disc_logits = disc(confusing_example)
    disc_loss = disc_loss_fn(disc_logits, y)
    disc_opt.zero_grad()
    disc_loss.backward()
    disc_opt.step()
    disc_acc = np.mean(
        np.equal(
            np.argmax(disc_logits.detach().cpu().numpy(), axis=-1),
            y.detach().cpu().numpy()
        )
    )
    
    rv = {}
    gen.train()
    gen_logits = gen(x)
    gen_adv_loss = gen_loss_fn(gen_logits, confusing_example)
    if reconstruction_critic_items is None:
        gen_rec_loss = gen_loss_fn(gen_logits, x)
    else:
        rc_disc, rc_opt = reconstruction_critic_items
        rc_disc.eval()
        gen_rec_loss = -torch.mean(rc_disc(gen_logits))
    if project_rec_updates:
        gen_opt.zero_grad()
        gen_adv_loss.backward(retain_graph=True)
        gen_adv_gradients = [param.grad.clone() for param in gen.parameters()]
        gen_opt.zero_grad()
        gen_rec_loss.backward()
        gen_rec_gradients = [param.grad.clone() for param in gen.parameters()]
        
        gen_proj_gradients = []
        for adv_grad, rec_grad in zip(gen_adv_gradients, gen_rec_gradients):
            dot = lambda x, y: (x*y).sum()
            proj_rec_grad = rec_grad - adv_grad*dot(adv_grad, rec_grad)/(dot(adv_grad, adv_grad)+1e-12)
            # project onto space orthogonal to the gradient of the adversarial loss, i.e. move in a
            #  direction which keeps this loss constant (to the extent that it is well-approximated
            #  by a hyperplane)
            proj_grad = loss_mixture_coefficient*adv_grad + (1-loss_mixture_coefficient)*proj_rec_grad
            gen_proj_gradients.append(proj_grad)
        for param, grad in zip(gen.parameters(), gen_proj_gradients):
            param.grad = grad
        gen_opt.step()
        
        rv.update({
            'gen_adv_loss': gen_adv_loss.detach().cpu().numpy(),
            'gen_rec_loss': gen_rec_loss.detach().cpu().numpy()
        })
        
    else:
        gen_loss = loss_mixture_coefficient*gen_adv_loss + (1-loss_mixture_coefficient)*gen_rec_loss
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()
        rv.update({
            'gen_adv_loss': gen_adv_loss.detach().cpu().numpy(),
            'gen_rec_loss': gen_rec_loss.detach().cpu().numpy()})
    
    if reconstruction_critic_items is not None:
        rc_disc.train()
        rc_logits_real = rc_disc(x)
        rc_logits_fake = rc_disc(gen_logits.detach())
        rc_loss_real = -torch.mean(rc_logits_real)
        rc_loss_fake = torch.mean(rc_logits_fake)
        rc_loss = rc_loss_real + rc_loss_fake
        rc_opt.zero_grad()
        rc_loss.backward()
        rc_opt.step()
        rv.update({
            'rc_loss_real': rc_loss_real.detach().cpu().numpy(),
            'rc_loss_fake': rc_loss_fake.detach().cpu().numpy()})
    
    rv.update({
        'confusing_example_loss': confusing_example_loss,
        'disc_loss': disc_loss.detach().cpu().numpy(),
        'disc_acc': disc_acc
    })
    return rv

def eval_step(batch, disc, gen, disc_loss_fn, gen_loss_fn, device,
              return_example=False,
              ce_kwargs={},
              loss_mixture_coefficient=0.5,
              reconstruction_critic_items=None):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    disc.eval()
    confusing_example, confusing_example_loss = compute_confusing_example(x, disc, gen, **ce_kwargs)
    
    disc_logits = disc(confusing_example)
    disc_loss = disc_loss_fn(disc_logits, y)
    disc_acc = np.mean(
        np.equal(
            np.argmax(disc_logits.detach().cpu().numpy(), axis=-1),
            y.detach().cpu().numpy()
        )
    )
    rv = {
        'confusing_example_loss': confusing_example_loss,
        'disc_loss': disc_loss.detach().cpu().numpy(),
        'disc_acc': disc_acc}
    
    gen.eval()
    gen_logits = gen(x).detach()
    gen_adv_loss = gen_loss_fn(gen_logits, confusing_example)
    if reconstruction_critic_items is None:
        gen_rec_loss = gen_loss_fn(gen_logits, x)
    else:
        rc_disc, _ = reconstruction_critic_items
        rc_disc.eval()
        rc_loss_fake = torch.mean(rc_disc(gen_logits))
        rc_loss_real = -torch.mean(rc_disc(x))
        gen_rec_loss = -rc_loss_fake
        rv.update({
            'rc_loss_real': rc_loss_real.detach().cpu().numpy(),
            'rc_loss_fake': rc_loss_fake.detach().cpu().numpy()})
    rv.update({
        'gen_adv_loss': gen_adv_loss.detach().cpu().numpy(),
        'gen_rec_loss': gen_rec_loss.detach().cpu().numpy()})
    
    if return_example:
        rv.update({
            'clean_example': x.cpu().numpy(),
            'confusing_example': to_uint8(confusing_example).cpu().numpy(),
            'generated_example': to_uint8(gen_logits.detach()).cpu().numpy()})
    return rv

def eval_evaldisc_step(batch, gen, eval_disc, eval_disc_loss_fn, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    gen.eval()
    eval_disc.eval()
    confusing_example = gen(x)
    confusing_example = to_uint8(confusing_example)
    eval_disc_logits = eval_disc(confusing_example)
    eval_disc_loss = eval_disc_loss_fn(eval_disc_logits, y)
    eval_disc_acc = np.mean(
        np.equal(
            np.argmax(eval_disc_logits.detach().cpu().numpy(), axis=-1),
            y.detach().cpu().numpy()
        )
    )
    
    return {
        'eval_disc_loss': eval_disc_loss.detach().cpu().numpy(),
        'eval_disc_acc': eval_disc_acc
    }

def train_epoch(dataloader, disc, gen, disc_opt, gen_opt, disc_loss_fn, gen_loss_fn, device,
                pretrain_disc_phase=False,
                pretrain_gen_phase=False,
                pretrain_eval_disc_phase=False,
                posttrain_eval_disc_phase=False,
                eval_disc_items=None,
                progress_bar=None,
                project_rec_updates=False,
                ce_kwargs={},
                loss_mixture_coefficient=0.5,
                reconstruction_critic_items=None,
                disc_orig_sample_prob=0.0):
    results = {}
    for batch in dataloader:
        if pretrain_disc_phase:
            rv = pretrain_disc_step(batch, disc, disc_opt, disc_loss_fn, device)
        elif pretrain_eval_disc_phase:
            assert eval_disc_items is not None
            eval_disc, eval_disc_opt, eval_disc_loss_fn = eval_disc_items
            rv = pretrain_disc_step(batch, eval_disc, eval_disc_opt, eval_disc_loss_fn, device)
        elif pretrain_gen_phase:
            rv = pretrain_gen_step(batch, gen, gen_opt, gen_loss_fn, device)
        elif posttrain_eval_disc_phase:
            assert eval_disc_items is not None
            eval_disc, eval_disc_opt, eval_disc_loss_fn = eval_disc_items
            rv = train_evaldisc_step(batch, gen, eval_disc, eval_disc_opt, eval_disc_loss_fn, device)
        else:
            rv = train_step(batch, disc, gen, disc_opt, gen_opt, disc_loss_fn, gen_loss_fn, device,
                            project_rec_updates=project_rec_updates,
                            ce_kwargs=ce_kwargs,
                            loss_mixture_coefficient=loss_mixture_coefficient,
                            reconstruction_critic_items=reconstruction_critic_items,
                            disc_orig_sample_prob=disc_orig_sample_prob)
            if eval_disc_items is not None:
                eval_disc, eval_disc_opt, eval_disc_loss_fn = eval_disc_items
                erv = train_evaldisc_step(batch, gen, eval_disc, eval_disc_opt, eval_disc_loss_fn, device)
                rv.update(erv)
        for key, item in rv.items():
            if pretrain_disc_phase or pretrain_gen_phase:
                key = 'pt_'+key
            elif pretrain_eval_disc_phase:
                key = 'pt_eval_'+key
            elif posttrain_eval_disc_phase:
                key = 'po_'+key
            if not key in results.keys():
                results[key] = []
            results[key].append(item)
        if progress_bar is not None:
            progress_bar.update(1)
    for key, item in results.items():
        results[key] = np.mean(item)
    return results

def eval_epoch(dataloader, disc, gen, disc_loss_fn, gen_loss_fn, device,
               return_example=False,
               pretrain_disc_phase=False,
               pretrain_gen_phase=False,
               pretrain_eval_disc_phase=False,
               posttrain_eval_disc_phase=False,
               eval_disc_items=None,
               progress_bar=None,
               ce_kwargs={},
               loss_mixture_coefficient=0.5,
               reconstruction_critic_items=None):
    results = {}
    for idx, batch in enumerate(dataloader):
        return_example_ = return_example and (idx == len(dataloader)-1)
        if pretrain_disc_phase:
            rv = eval_disc_step(batch, disc, disc_loss_fn, device)
        elif pretrain_eval_disc_phase:
            assert eval_disc_items is not None
            eval_disc, eval_disc_opt, eval_disc_loss_fn = eval_disc_items
            rv = pretrain_disc_step(batch, eval_disc, eval_disc_opt, eval_disc_loss_fn, device)
        elif pretrain_gen_phase:
            rv = eval_gen_step(batch, gen, gen_loss_fn, device)
        elif posttrain_eval_disc_phase:
            assert eval_disc_items is not None
            eval_disc, eval_disc_opt, eval_disc_loss_fn = eval_disc_items
            rv = eval_evaldisc_step(batch, gen, eval_disc, eval_disc_loss_fn, device)
        else:
            rv = eval_step(batch, disc, gen, disc_loss_fn, gen_loss_fn, device,
                           return_example=return_example_,
                           ce_kwargs=ce_kwargs,
                           loss_mixture_coefficient=loss_mixture_coefficient,
                           reconstruction_critic_items=reconstruction_critic_items)
            if eval_disc_items is not None:
                eval_disc, eval_disc_opt, eval_disc_loss_fn = eval_disc_items
                erv = eval_evaldisc_step(batch, gen, eval_disc, eval_disc_loss_fn, device)
                rv.update(erv)
        if return_example_:
            results['clean_example'] = rv['clean_example']
            results['confusing_example'] = rv['confusing_example']
            results['generated_example'] = rv['generated_example']
            del rv['clean_example']
            del rv['confusing_example']
            del rv['generated_example']
        for key, item in rv.items():
            if pretrain_disc_phase or pretrain_gen_phase:
                key = 'pt_'+key
            elif pretrain_eval_disc_phase:
                key = 'pt_eval_'+key
            elif posttrain_eval_disc_phase:
                key = 'po_'+key
            if not key in results.keys():
                results[key] = []
            results[key].append(item)
        if progress_bar is not None:
            progress_bar.update(1)
    for key, item in results.items():
        if key not in ['clean_example', 'confusing_example', 'generated_example']:
            results[key] = np.mean(item)
    return results