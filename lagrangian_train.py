import numpy as np
import torch
from torch import nn, optim

class Encoder(nn.Module):
    def __init__(self, use_sn=False):
        super().__init__()
        
        sn = nn.utils.spectral_norm if use_sn else lambda x: x
        
        self.feature_extractor = nn.Sequential(
            sn(nn.Conv2d(1, 8, 3, stride=1, padding=1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            sn(nn.Conv2d(8, 32, 3, stride=1, padding=1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
        
        self.shortcut = nn.Sequential(
            sn(nn.Linear(1*28*28, 1*7*7)),
            sn(nn.Linear(1*7*7, 32*7*7))
        )
        
        self.feature_mixer = nn.Sequential(
            sn(nn.Linear(32*7*7, 256)),
            nn.LeakyReLU(0.2),
            sn(nn.Linear(256, 128))
        )
        
    def forward(self, x):
        x_features = self.feature_extractor(x)
        x_features = x_features.view(-1, np.prod(x_features.shape[1:]))
        shortcut_features = self.shortcut(x.view(-1, np.prod(x.shape[1:])))
        features = x_features + shortcut_features
        x_mixed_features = self.feature_mixer(features)
        return x_mixed_features

class Decoder(nn.Module):
    def __init__(self, output_transform=nn.Hardtanh, use_sn=False):
        super().__init__()
        
        sn = nn.utils.spectral_norm if use_sn else lambda x: x
        
        self.feature_mixer = nn.Sequential(
            nn.LeakyReLU(0.2),
            sn(nn.Linear(128, 256)),
            nn.LeakyReLU(0.2),
            sn(nn.Linear(256, 32*7*7)),
        )
        
        self.reconstructor = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            sn(nn.ConvTranspose2d(32, 8, 3, stride=1, padding=1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            sn(nn.ConvTranspose2d(8, 1, 3, stride=1, padding=1))
        )
        
        self.output_transform = output_transform()
        
    def forward(self, x):
        mixed_features = self.feature_mixer(x)
        mixed_features = mixed_features.view(-1, 32, 7, 7)
        reconstructed_logits = self.reconstructor(mixed_features)
        output = self.output_transform(reconstructed_logits)
        return output

class Classifier(nn.Module):
    def __init__(self, use_sn=False, num_classes=2):
        super().__init__()
        
        sn = nn.utils.spectral_norm if use_sn else lambda x: x
        
        self.classifier = nn.Sequential(
            sn(nn.Linear(128, 512)),
            nn.LayerNorm((512,)),
            nn.LeakyReLU(0.2),
            sn(nn.Linear(512, 512)),
            nn.LayerNorm((512,)),
            nn.LeakyReLU(0.2),
            sn(nn.Linear(512, num_classes))
        )
    
    def forward(self, x):
        logits = self.classifier(x)
        return logits
    
class IndClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes))
    
    def forward(self, x):
        x_features = self.feature_extractor(x).view(-1, 256)
        logits = self.classifier(x_features)
        return logits

def confusion_loss_fn(logits, tolerance=0.0):
    uniform_dist = nn.functional.softmax(torch.zeros_like(logits), dim=-1)
    model_log_dist = nn.functional.log_softmax(logits, dim=-1)
    uniform_log_dist = nn.functional.log_softmax(torch.zeros_like(logits), dim=-1)
    kl_div = (uniform_dist*(uniform_log_dist-model_log_dist)).sum()
    if kl_div < tolerance:
        kl_div = 0.0*kl_div
    return kl_div

def val(tensor):
    return tensor.detach().cpu().numpy()

def acc(logits, labels):
    logits, labels = val(logits), val(labels)
    acc = np.mean(np.equal(np.argmax(logits, axis=-1), labels))
    return acc

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item()**2
    total_norm = total_norm**0.5
    return total_norm

def to_uint8(x):
    x = 255.0*(0.5*x+0.5)
    x = x.to(torch.uint8)
    x = x.to(torch.float)
    x = 2.0*(x/255.0)-1.0
    return x

def get_models(enc_constructor, enc_kwargs,
               dec_constructor, dec_kwargs,
               cls_constructor, cls_kwargs,
               icls_constructor, icls_kwargs,
               enc_opt_constructor, enc_opt_kwargs,
               dec_opt_constructor, dec_opt_kwargs,
               lbd_opt_constructor, lbd_opt_kwargs,
               cls_opt_constructor, cls_opt_kwargs,
               icls_opt_constructor, icls_opt_kwargs,
               cls_loss_fn_constructor, cls_loss_fn_kwargs,
               rec_loss_fn_constructor, rec_loss_fn_kwargs,
               icls_loss_fn_constructor, icls_loss_fn_kwargs,
               device, input_shape, num_classes):
    enc = enc_constructor(input_shape=input_shape, **enc_kwargs).to(device)
    enc_opt = enc_opt_constructor(enc.parameters(), **enc_opt_kwargs)
    dec = dec_constructor(input_shape=input_shape, **dec_kwargs).to(device)
    dec_opt = dec_opt_constructor(dec.parameters(), **dec_opt_kwargs)
    lbd = torch.tensor(0.0, dtype=torch.float, device=device, requires_grad=True)
    lbd_opt = lbd_opt_constructor([lbd], **lbd_opt_kwargs)
    cls = cls_constructor(num_classes=num_classes, **cls_kwargs).to(device)
    cls_opt = cls_opt_constructor(cls.parameters(), **cls_opt_kwargs)
    icls = icls_constructor(num_classes=num_classes, **icls_kwargs).to(device)
    icls_opt = icls_opt_constructor(icls.parameters(), **icls_opt_kwargs)
    cls_loss_fn = cls_loss_fn_constructor(**cls_loss_fn_kwargs).to(device)
    rec_loss_fn = rec_loss_fn_constructor(**rec_loss_fn_kwargs).to(device)
    icls_loss_fn = icls_loss_fn_constructor(**icls_loss_fn_kwargs).to(device)
    train_args = (enc, enc_opt, dec, dec_opt, lbd, lbd_opt, cls, cls_opt, cls_loss_fn, rec_loss_fn, device)
    eval_args = (enc, dec, lbd, cls, cls_loss_fn, rec_loss_fn, device)
    itrain_args = (enc, dec, icls, icls_opt, icls_loss_fn, device)
    ieval_args = (enc, dec, icls, icls_loss_fn, device)
    return train_args, eval_args, itrain_args, ieval_args

def train_ind_cls_step(batch, enc, dec, ind_cls, ind_cls_opt, ind_cls_loss_fn, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    enc.eval()
    dec.eval()
    ind_cls.train()
    
    with torch.no_grad():
        x_enc = enc(x)
        x_rec = dec(x_enc)
        x_rec = to_uint8(x_rec)
    ind_cls_logits = ind_cls(x_rec)
    ind_cls_loss = ind_cls_loss_fn(ind_cls_logits, y)
    ind_cls_opt.zero_grad()
    ind_cls_loss.backward()
    ind_cls_opt.step()
    
    rv = {
        'loss': val(ind_cls_loss),
        'acc': acc(ind_cls_logits, y)
    }
    return rv

@torch.no_grad()
def eval_ind_cls_step(batch, enc, dec, ind_cls, ind_cls_loss_fn, device):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    enc.eval()
    dec.eval()
    ind_cls.eval()
    
    x_enc = enc(x)
    x_rec = dec(x_enc)
    x_rec = to_uint8(x_rec)
    ind_cls_logits = ind_cls(x_rec)
    ind_cls_loss = ind_cls_loss_fn(ind_cls_logits, y)
    
    rv = {
        'loss': val(ind_cls_loss),
        'acc': acc(ind_cls_logits, y)
    }
    return rv

def train_step(batch, enc, enc_opt, dec, dec_opt, lbd, lbd_opt, cls, cls_opt, cls_loss_fn, rec_loss_fn, device,
               return_example=False, compute_grad_norms=False, lambda_clamp_val=None, cls_steps_per_enc_step=1.0):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    enc.train()
    dec.train()
    cls.train()
    
    step_cls = cls_steps_per_enc_step>=1.0 or np.random.uniform(0, 1)<cls_steps_per_enc_step
    step_enc = cls_steps_per_enc_step<=1.0 or np.random.uniform(0, 1)<1/cls_steps_per_enc_step
    
    # Update encoder, decoder, lambda
    x_enc = enc(x)
    x_rec = dec(x_enc)
    cls_logits = cls(x_enc)
    rec_loss = rec_loss_fn(x_rec, x)
    conf_loss = confusion_loss_fn(cls_logits)
    if lambda_clamp_val is not None:
        mixture_coefficient = lambda_clamp_val
    else:
        mixture_coefficient = lbd
    lagrangian = rec_loss + mixture_coefficient*conf_loss
    
    if step_enc:
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        if lambda_clamp_val is None:
            lbd_opt.zero_grad()
        lagrangian.backward()
        if compute_grad_norms:
            enc_grad_norm = grad_norm(enc)
            dec_grad_norm = grad_norm(dec)
            if lambda_clamp_val is None:
                lbd_grad_norm = np.abs(lbd.grad.detach().cpu().numpy())
        enc_opt.step()
        dec_opt.step()
        if lambda_clamp_val is None:
            lbd.grad = lbd.grad*-1.0
            lbd_opt.step()
    else:
        enc_grad_norm = dec_grad_norm = lbd_grad_norm = 0.0
    
    # Update classifier
    cls_logits = cls(x_enc.detach()) # We can probably avoid recomputing this
    cls_loss = cls_loss_fn(cls_logits, y)
    if step_cls:
        cls_opt.zero_grad()
        cls_loss.backward()
        if compute_grad_norms:
            cls_grad_norm = grad_norm(cls)
        cls_opt.step()
    else:
        cls_grad_norm = 0.0
    
    rv = {
        'lagrangian': val(lagrangian),
        'lambda': val(lbd),
        'rec_loss': val(rec_loss),
        'conf_loss': val(conf_loss),
        'cls_loss': val(cls_loss),
        'cls_acc': acc(cls_logits, y)
    }
    if compute_grad_norms:
        rv.update({
            'enc_grad_norm': enc_grad_norm,
            'dec_grad_norm': dec_grad_norm,
            'lbd_grad_norm': 0.0 if lambda_clamp_val is not None else lbd_grad_norm,
            'cls_grad_norm': cls_grad_norm
        })
    if return_example:
        orig_example = val(to_uint8(x))
        rec_example = val(to_uint8(x_rec))
        rv.update({
            'orig_example': 0.5*orig_example+0.5,
            'rec_example': 0.5*rec_example+0.5
        })
    return rv

@torch.no_grad()
def eval_step(batch, enc, dec, lbd, cls, cls_loss_fn, rec_loss_fn, device,
              return_example=False,
              lambda_clamp_val=None):
    x, y, _ = batch
    x, y = x.to(device), y.to(device)
    enc.eval()
    dec.eval()
    cls.eval()
    
    x_enc = enc(x)
    x_rec = dec(x_enc)
    cls_logits = cls(x_enc)
    rec_loss = rec_loss_fn(x_rec, x)
    conf_loss = confusion_loss_fn(cls_logits)
    if lambda_clamp_val is None:
        lagrangian = rec_loss + lbd*conf_loss
    else:
        lagrangian = rec_loss + lambda_clamp_val*conf_loss
    cls_loss = cls_loss_fn(cls_logits, y)
    
    rv = {
        'lagrangian': val(lagrangian),
        'rec_loss': val(rec_loss),
        'conf_loss': val(conf_loss),
        'cls_loss': val(cls_loss),
        'cls_acc': acc(cls_logits, y)
    }
    if return_example:
        orig_example = val(to_uint8(x))
        rec_example = val(to_uint8(x_rec))
        rv.update({
            'orig_example': 0.5*orig_example+0.5,
            'rec_example': 0.5*rec_example+0.5
        })
    return rv

def train_epoch(dataloader, *step_args, return_example_idx=None, compute_grad_norms=False, lambda_clamp_val=None, indcls=False, cls_steps_per_enc_step=1.0, separate_cls_partition=False):
    rv = {}
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx)==int else return_example_idx
    if separate_cls_partition:
        dataloader = zip(*dataloader)
    for bidx, batch in enumerate(dataloader):
        if indcls:
            step_rv = train_ind_cls_step(batch, *step_args)
        else:
            if separate_cls_partition:
                enc_batch, cls_batch = batch
                step_rv = train_step(enc_batch, *step_args,
                                     return_example=bidx in re_indices,
                                     compute_grad_norms=compute_grad_norms,
                                     lambda_clamp_val=lambda_clamp_val,
                                     cls_steps_per_enc_step=0.0)
                cls_rv = train_step(cls_batch, *step_args,
                                         return_example=False,
                                         compute_grad_norms=compute_grad_norms,
                                         lambda_clamp_val=lambda_clamp_val,
                                         cls_steps_per_enc_step=np.inf)
                step_rv.update({
                    'cls_loss': cls_rv['cls_loss'],
                    'cls_acc': cls_rv['cls_acc'],
                    'cls_grad_norm': cls_rv['cls_grad_norm']
                })
            else:
                step_rv = train_step(batch, *step_args, 
                                     return_example=bidx in re_indices, 
                                     compute_grad_norms=compute_grad_norms, 
                                     lambda_clamp_val=lambda_clamp_val,
                                     cls_steps_per_enc_step=cls_steps_per_enc_step)
        for key, item in step_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    for key, item in rv.items():
        if not key in ['orig_example', 'rec_example']:
            rv[key] = np.mean(item)
    return rv

def eval_epoch(dataloader, *step_args, return_example_idx=None, lambda_clamp_val=None, indcls=False):
    rv = {}
    re_indices = [] if return_example_idx is None else [return_example_idx] if type(return_example_idx)==int else return_example_idx
    for bidx, batch in enumerate(dataloader):
        if indcls:
            step_rv = eval_ind_cls_step(batch, *step_args)
        else:
            step_rv = eval_step(batch, *step_args,
                                return_example=bidx in re_indices,
                                lambda_clamp_val=lambda_clamp_val)
        for key, item in step_rv.items():
            if not key in rv.keys():
                rv[key] = []
            rv[key].append(item)
    for key, item in rv.items():
        if not key in ['orig_example', 'rec_example']:
            rv[key] = np.mean(item)
    return rv