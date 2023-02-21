import numpy as np
import torch
from torch import nn

def get_sanitized_image(sanitizer, image):
    return nn.functional.hardtanh(sanitizer(image))

def val(x):
    return x.detach().cpu().numpy()

def get_acc(logits, target):
    logits = val(logits)
    target = val(target)
    return np.mean(np.equal(np.argmax(logits, axis=-1), target))

def get_oracle_loss(orig_image, sanitized_image):
    return val(nn.functional.mse_loss(orig_image, sanitized_image))

def train_autoencoder(dataloader, model, loss_fn, opt, device, n_epochs=10):
    model.train()
    init_loss = []
    for batch in dataloader:
        image, _, _ = batch
        image = image.to(device)
        reconstructed_image = model(image)
        loss = loss_fn(reconstructed_image, image)
        init_loss.append(val(loss))
    init_loss = np.mean(init_loss)
    for _ in range(n_epochs):
        for batch in dataloader:
            image, _, _ = batch
            image = image.to(device)
            reconstructed_image = model(image)
            loss = loss_fn(reconstructed_image, image)
            opt.zero_grad()
            loss.backward()
            opt.step()
    final_loss = []
    for batch in dataloader:
        image, _, _ = batch
        image = image.to(device)
        reconstructed_image = model(image)
        loss = loss_fn(reconstructed_image, image)
        final_loss.append(val(loss))
    final_loss = np.mean(final_loss)
    model.eval()
    return {'init_loss': init_loss, 'final_loss': final_loss}

def pretrain_c_step(batches, classifier, sanitizer, c_loss_fn, s_loss_fn, c_opt, s_opt, device, s_lbd=1.0):
    classifier.train()
    sanitizer.train()
    c_batch, s_batch = batches
    
    # step classifier
    image, watermark_target, metadata = c_batch
    image, watermark_target = image.to(device), watermark_target.to(device)
    sanitized_image = get_sanitized_image(sanitizer, image).detach()
    c_logits = classifier(sanitized_image)
    c_loss = c_loss_fn(c_logits, watermark_target)
    c_opt.zero_grad()
    c_loss.backward()
    c_opt.step()
    with torch.no_grad():
        orig_image = metadata['orig_image'].to(device)
        c_oracle_loss = get_oracle_loss(orig_image, sanitized_image)
    
    # eval sanitizer
    with torch.no_grad():
        image, watermark_target, metadata = s_batch
        image, watermark_target = image.to(device), watermark_target.to(device)
        sanitized_image = get_sanitized_image(sanitizer, image).detach()
        c_logits = classifier(sanitized_image)
        s_rec_loss = s_loss_fn.rec_loss(image, watermark_target, c_logits, sanitized_image)
        s_em_loss = s_loss_fn.em_loss(image, watermark_target, c_logits, sanitized_image)
        s_loss = s_em_loss + s_lbd*s_rec_loss
        orig_image = metadata['orig_image'].to(device)
        s_oracle_loss = get_oracle_loss(orig_image, sanitized_image)
    
    # return results
    return {
        'c_loss': val(c_loss),
        's_em_loss': val(s_em_loss),
        's_rec_loss': val(s_rec_loss),
        'c_acc': get_acc(c_logits, watermark_target),
        'c_oracle_loss': c_oracle_loss,
        's_oracle_loss': s_oracle_loss
    }

def pretrain_s_step(batches, classifier, sanitizer, c_loss_fn, s_loss_fn, c_opt, s_opt, device, s_lbd=0.0):
    classifier.train()
    sanitizer.train()
    c_batch, s_batch = batches
    
    # step sanitizer
    image, watermark_target, metadata = s_batch
    image, watermark_target = image.to(device), watermark_target.to(device)
    sanitized_image = get_sanitized_image(sanitizer, image)
    c_logits = classifier(sanitized_image)
    s_rec_loss = s_loss_fn.rec_loss(image, watermark_target, c_logits, sanitized_image)
    s_em_loss = s_loss_fn.em_loss(image, watermark_target, c_logits, sanitized_image)
    s_loss = s_lbd*s_rec_loss
    s_opt.zero_grad()
    s_loss.backward()
    s_opt.step()
    with torch.no_grad():
        orig_image = metadata['orig_image'].to(device)
        s_oracle_loss = get_oracle_loss(orig_image, sanitized_image.detach())
    
    # eval classifier
    with torch.no_grad():
        image, watermark_target, metadata = c_batch
        image, watermark_target = image.to(device), watermark_target.to(device)
        sanitized_image = get_sanitized_image(sanitizer, image).detach()
        c_logits = classifier(sanitized_image)
        c_loss = c_loss_fn(c_logits, watermark_target)
        orig_image = metadata['orig_image'].to(device)
        c_oracle_loss = get_oracle_loss(orig_image, sanitized_image)
    
    # return results
    return {
        'c_loss': val(c_loss),
        's_em_loss': val(s_em_loss),
        's_rec_loss': val(s_rec_loss),
        'c_acc': get_acc(c_logits, watermark_target),
        'c_oracle_loss': c_oracle_loss,
        's_oracle_loss': s_oracle_loss
    }

def train_step(batches, classifier, sanitizer, c_loss_fn, s_loss_fn, c_opt, s_opt, device, s_lbd=1.0):
    classifier.train()
    sanitizer.train()
    c_batch, s_batch = batches
    
    # step classifier
    image, watermark_target, metadata = c_batch
    image, watermark_target = image.to(device), watermark_target.to(device)
    sanitized_image = get_sanitized_image(sanitizer, image).detach()
    c_logits = classifier(sanitized_image)
    c_loss = c_loss_fn(c_logits, watermark_target)
    c_opt.zero_grad()
    c_loss.backward()
    c_opt.step()
    orig_image = metadata['orig_image'].to(device)
    c_oracle_loss = get_oracle_loss(orig_image, sanitized_image)
    
    # step sanitizer
    image, watermark_target, metadata = s_batch
    image, watermark_target = image.to(device), watermark_target.to(device)
    sanitized_image = get_sanitized_image(sanitizer, image)
    c_logits = classifier(sanitized_image)
    s_rec_loss = s_loss_fn.rec_loss(image, watermark_target, c_logits, sanitized_image)
    s_em_loss = s_loss_fn.em_loss(image, watermark_target, c_logits, sanitized_image)
    s_loss = s_em_loss + s_lbd*s_rec_loss
    s_opt.zero_grad()
    s_loss.backward()
    s_opt.step()
    orig_image = metadata['orig_image'].to(device)
    s_oracle_loss = get_oracle_loss(orig_image, sanitized_image)
    
    # return results
    return {
        'c_loss': val(c_loss),
        's_em_loss': val(s_em_loss),
        's_rec_loss': val(s_rec_loss),
        'c_acc': get_acc(c_logits, watermark_target),
        'c_oracle_loss': c_oracle_loss,
        's_oracle_loss': s_oracle_loss
    }

@torch.no_grad()
def eval_step(batches, classifier, sanitizer, c_loss_fn, s_loss_fn, device, s_lbd=1.0):
    classifier.eval()
    sanitizer.eval()
    if type(batches) == tuple:
        c_batch, s_batch = batches
    else:
        c_batch = s_batch = batches
    
    # eval classifier
    image, watermark_target, metadata = c_batch
    image, watermark_target = image.to(device), watermark_target.to(device)
    sanitized_image = get_sanitized_image(sanitizer, image).detach()
    c_logits = classifier(sanitized_image)
    c_loss = c_loss_fn(c_logits, watermark_target)
    orig_image = metadata['orig_image'].to(device)
    c_oracle_loss = get_oracle_loss(orig_image, sanitized_image)
    
    # eval sanitizer
    image, watermark_target, metadata = s_batch
    image, watermark_target = image.to(device), watermark_target.to(device)
    sanitized_image = get_sanitized_image(sanitizer, image).detach()
    c_logits = classifier(sanitized_image)
    s_em_loss = s_loss_fn.em_loss(image, watermark_target, c_logits, sanitized_image)
    s_rec_loss = s_loss_fn.rec_loss(image, watermark_target, c_logits, sanitized_image)
    orig_image = metadata['orig_image'].to(device)
    s_oracle_loss = get_oracle_loss(orig_image, sanitized_image)
    
    # return results
    return {
        'c_loss': val(c_loss),
        's_em_loss': val(s_em_loss),
        's_rec_loss': val(s_rec_loss),
        'c_acc': get_acc(c_logits, watermark_target),
        'c_oracle_loss': c_oracle_loss,
        's_oracle_loss': s_oracle_loss
    }

def run_epoch(dataloader, step_fn, args, pref='', s_lbd=1.0):
    results = {}
    if type(dataloader) == list:
        dataloader = zip(*dataloader)
    for batch in dataloader:
        rv = step_fn(batch, *args, s_lbd=s_lbd)
        for key in rv.keys():
            if not pref+key in results.keys():
                results[pref+key] = []
            results[pref+key].append(rv[key])
    for key, item in results.items():
        results[key] = np.mean(item)
    return results