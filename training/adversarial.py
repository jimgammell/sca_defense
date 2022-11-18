import torch
from torch import nn
from training.common import to_np, accuracy, mean_rank, local_avg, unpack_batch

def adversarial_train_step(batch, disc, disc_loss_fn, disc_opt, gen, gen_loss_fn, gen_opt, device, grad_clip=None):
    trace, label = unpack_batch(batch, device)
    disc.train()
    gen.train()
    masking_trace = gen(trace)
    disc_logits = disc(trace+masking_trace)
    gen_loss = gen_loss_fn(disc_logits, label)
    gen_opt.zero_grad()
    gen_loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(gen.parameters(), max_norm=grad_clip, norm_type=2)
    gen_opt.step()
    disc_logits = disc(trace+masking_trace.detach())
    disc_loss = disc_loss_fn(disc_logits, label)
    disc_opt.zero_grad()
    disc_loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(disc.parameters(), max_norm=grad_clip, norm_type=2)
    disc_opt.step()
    return {'gen_loss': to_np(gen_loss),
            'disc_loss': to_np(disc_loss),
            'disc_acc': accuracy(disc_logits, label),
            'disc_mean_rank': mean_rank(disc_logits, label)}

@torch.no_grad()
def adversarial_eval_step(batch, disc, disc_loss_fn, gen, gen_loss_fn, device):
    trace, label = unpack_batch(batch, device)
    disc.eval()
    gen.eval()
    masking_trace = gen(trace)
    disc_logits = disc(trace+masking_trace)
    gen_loss = gen_loss_fn(disc_logits, label)
    disc_loss = disc_loss_fn(disc_logits, label)
    return {'gen_loss': to_np(gen_loss),
            'disc_loss': to_np(disc_loss),
            'disc_acc': accuracy(disc_logits, label),
            'disc_mean_rank': mean_rank(disc_logits, label)}