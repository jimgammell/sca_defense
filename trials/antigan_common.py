

from utils import get_print_to_log, get_filename
print = get_print_to_log(get_filename(__file__))
from trial_common import clamp_model_params, extract_rv_from_tensor, get_model_params_histogram

class AntiGanExperiment:
    def __init__(self,
                 disc,
                 gen,
                 disc_loss_fn,
                 gen_loss_fn,
                 disc_opt,
                 gen_opt,
                 device,
                 latent_vars_distr=None,
                 conditions=[],
                 use_labels=False,
                 objective_formulation='Complement'):
        pass
    
    def train_step(self, batch, train_disc=True, train_gen=True):
        self.disc.train()
        self.gen.train()
        
        raw_images, labels = batch
        raw_images = raw_images.to(self.device)
        raw_labels = raw_labels.to(self.device)
        
        def get_disc_logits():
            gen_args = []
            if self.use_latent_variables:
                latent_variables = self.latent_vars_distr.sample().to(self.device)
                gen_args.append(latent_variables)
            if self.use_labels:
                gen_args.append(raw_labels)
            if self.use_conditions:
                conditions = self.get_conditions(raw_images.size(0))
            protective_noise = self.gen(*gen_args)
            protected_images = raw_images + protective_noise
            disc_logits = self.disc(protected_images)
        
        if train_gen:
            disc_logits = get_disc_logits()
            gen_loss = self.get_gen_loss(disc_logits, labels)
            self.gen_opt.zero_grad()
            gen_loss.backward()
            self.gen_opt.step()
            clamp_model_params(self.gen)
            
        if train_disc:
            disc_logits = get_disc_logits()
            disc_loss = self.get_disc_loss(disc_logits, labels)
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            clamp_model_params(self.disc)
        