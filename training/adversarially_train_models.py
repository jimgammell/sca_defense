import torch
from torch import nn
from training.common import unpack_batch, detach_result, run_epoch

def get_results(disc_loss, gen_loss, disc_logits, gen_logits, x, y, disc_metric_fns, gen_metric_fns):
    if disc_metric_fns is None:
        disc_metric_fns = {}
    if gen_metric_fns is None:
        gen_metric_fns = {}
    results = {
        'disc_loss': detach_result(disc_loss),
        'gen_loss': detach_result(gen_loss),
        **{'disc_'+metric_name: metric_fn(x, y, disc_logits) for metric_name, metric_fn in disc_metric_fns.items()},
        **{'gen_'+metric_name: metric_fn(x, y, gen_logits) for metric_name, metric_fn in gen_metric_fns.items()}
    }
    return results

def train_step(batch, disc, disc_loss_fn, disc_opt, gen, gen_loss_fn, gen_opt, device,
               disc_grad_clip=None, gen_grad_clip=None, disc_metric_fns=None, gen_metric_fns=None,
               step_gen=True, step_disc=True):
    x, y = unpack_batch(batch, device)
    disc.train()
    gen.train()
    gen_logits = gen(x)
    disc_logits = disc(x+gen_logits)
    disc_loss = disc_loss_fn(disc_logits, x, y)
    if step_disc:
        disc_opt.zero_grad()
        disc_loss.backward()
        if type(disc_grad_clip) == float:
            nn.utils.clip_grad_norm_(disc.parameters(), max_norm=disc_grad_clip, norm_type=2)
        disc_opt.step()
    
    ## Probably don't need to recompute both logits
    gen_logits = gen(x)
    disc_logits = disc(x+gen_logits)
    gen_loss = gen_loss_fn(disc_logits, gen_logits, x, y)
    if step_gen:
        gen_opt.zero_grad()
        gen_loss.backward()
        if type(gen_grad_clip) == float:
            nn.utils.clip_grad_norm_(gen.parameters(), max_norm=gen_grad_clip, norm_type=2)
        gen_opt.step()
    results = get_results(disc_loss, gen_loss, disc_logits, gen_logits, x, y, disc_metric_fns, gen_metric_fns)
    return results

@torch.no_grad()
def eval_step(batch, disc, disc_loss_fn, gen, gen_loss_fn, device, disc_metric_fns=None, gen_metric_fns=None):
    x, y = unpack_batch(batch, device)
    disc.eval()
    gen.eval()
    gen_logits = gen(x)
    disc_logits = disc(x+gen_logits)
    disc_loss = disc_loss_fn(disc_logits, x, y)
    gen_loss = gen_loss_fn(disc_logits, gen_logits, x, y)
    results = get_results(disc_loss, gen_loss, disc_logits, gen_logits, x, y, disc_metric_fns, gen_metric_fns)
    return results

def train_epoch(dataloader, disc, disc_loss_fn, disc_opt, gen, gen_loss_fn, gen_opt, device,
                disc_grad_clip=None, gen_grad_clip=None, disc_metric_fns=None, gen_metric_fns=None,
                disc_steps_per_gen_step=1.0, **kwargs):
    results = {}
    disc_steps, gen_steps = 1.0, 1.0
    for batch in dataloader:
        step_disc = disc_steps/gen_steps <= disc_steps_per_gen_step
        step_gen = disc_steps/gen_steps >= disc_steps_per_gen_step
        rv = train_step(
            batch, disc, disc_loss_fn, disc_opt, gen, gen_loss_fn, gen_opt, device,
            disc_grad_clip=disc_grad_clip, gen_grad_clip=gen_grad_clip,
            disc_metric_fns=disc_metric_fns, gen_metric_fns=gen_metric_fns,
            step_gen=step_gen, step_disc=step_disc
        )
        disc_steps += 1.0 if step_disc else 0.0
        gen_steps += 1.0 if step_gen else 0.0
        for key, item in rv.items():
            if not key in results.keys():
                results[key] = []
            results[key].append(item)
    return results

def eval_epoch(dataloader, disc, disc_loss_fn, gen, gen_loss_fn, device,
               disc_metric_fns=None, gen_metric_fns=None, **kwargs):
    results = run_epoch(eval_step, dataloader, disc, disc_loss_fn, gen, gen_loss_fn, device,
                        disc_metric_fns=disc_metric_fns, gen_metric_fns=gen_metric_fns, **kwargs)
    if 'decision_boundary_frame' in disc_metric_fns.keys():
        disc_metric_fns['decision_boundary_frame'](disc=disc, gen=gen, dataset=dataloader.dataset)
    return results