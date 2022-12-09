import numpy as np
import torch
from torch import nn
from training.common import to_np, accuracy, mean_rank, local_avg, unpack_batch

class LstmModel(nn.Module):
    def __init__(self,
                 encoder_layers=[256],
                 mask_decoder_layers=[256],
                 pred_decoder_layers=[256],
                 pred_classes=256,
                 input_features=1,
                 output_features=1,
                 delay=1):
        super().__init__()
        
        self.encoder_layers = [input_features] + encoder_layers
        self.mask_decoder_layers = [self.encoder_layers[-1]] + mask_decoder_layers
        self.pred_decoder_layers = [self.encoder_layers[-1]] + pred_decoder_layers
        self.pred_classes = pred_classes
        self.input_features = input_features
        self.output_features = output_features
        self.delay = delay
        
        self.encoder = nn.ModuleList([
            nn.GRUCell(input_size=l1, hidden_size=l2)
            for l1, l2 in zip(self.encoder_layers[:-1], self.encoder_layers[1:])])
        self.mask_decoder = nn.ModuleList([
            nn.GRUCell(input_size=l1, hidden_size=l2)
            for l1, l2 in zip(self.mask_decoder_layers[:-1], self.mask_decoder_layers[1:])] +
            [nn.Linear(self.mask_decoder_layers[-1], output_features)])
        self.pred_decoder = nn.ModuleList([
            nn.GRUCell(input_size=l1, hidden_size=l2)
            for l1, l2 in zip(self.pred_decoder_layers[:-1], self.pred_decoder_layers[1:])] +
            [nn.Linear(self.pred_decoder_layers[-1], pred_classes)])
    
    def init_state(self, x):
        self.encoder_state = [torch.randn(x.shape[1], width, device=x.device)
                               for width in self.encoder_layers[1:]]
        self.mask_decoder_state = [torch.randn(x.shape[1], width, device=x.device)
                                   for width in self.mask_decoder_layers[1:]]
        self.pred_decoder_state = [torch.randn(x.shape[1], width, device=x.device)
                                   for width in self.pred_decoder_layers[1:]]
    
    def forward_timestep(self, x):
        for idx, (prev_state, pres_state, layer) in enumerate(zip([x]+self.encoder_state[:-1], self.encoder_state, self.encoder)):
            new_pres_state_enc = layer(prev_state, pres_state)
            self.encoder_state[idx] = new_pres_state_enc
        enc_x = new_pres_state_enc
        for idx, (prev_state, pres_state, layer) in enumerate(zip([enc_x]+self.mask_decoder_state[:-1], self.mask_decoder_state, self.mask_decoder[:-1])):
            new_pres_state_mdec = layer(prev_state, pres_state)
            self.mask_decoder_state[idx] = new_pres_state_mdec
        mask = self.mask_decoder[-1](new_pres_state_mdec)
        for idx, (prev_state, pres_state, layer) in enumerate(zip([enc_x]+self.pred_decoder_state[:-1], self.pred_decoder_state, self.pred_decoder[:-1])):
            new_pres_state_pdec = layer(prev_state, pres_state)
            self.pred_decoder_state[idx] = new_pres_state_pdec
        pred = self.pred_decoder[-1](new_pres_state_pdec)
        return mask, pred
    
    def forward(self, sequence, start_diff=None, end_diff=None):
        if start_diff == None:
            start_diff = 0
        if end_diff == None:
            end_diff = len(sequence)
        mask_values, pred_values = [], []
        self.init_state(sequence)
        delay_padding = torch.randn(self.delay, *sequence.shape[1:], device=sequence.device)
        sequence = torch.cat((delay_padding, sequence[:-self.delay, :, :]))
        with torch.no_grad():
            for x in sequence[:start_diff]:
                mask_value, pred_value = self.forward_timestep(x)# if len(mask_values) == 0 else x+mask_values[-1])
                mask_values.append(mask_value)
                pred_values.append(pred_value)
        for x in sequence[start_diff:end_diff]:
            mask_value, pred_value = self.forward_timestep(x)# if len(mask_values) == 0 else x+mask_values[-1])
            mask_values.append(mask_value)
            pred_values.append(pred_value)
        with torch.no_grad():
            for x in sequence[end_diff:]:
                mask_value, pred_value = self.forward_timestep(x)# if len(mask_values) == 0 else x+mask_values[-1])
                mask_values.append(mask_value)
                pred_values.append(pred_value)
        mask = torch.stack(mask_values).view(*sequence.shape)
        preds = torch.mean(torch.stack(pred_values), dim=0).view(sequence.shape[1], self.pred_classes)
        return mask, preds

def pretrain_step(batch, model, mask_loss_fn, pred_loss_fn, optimizer, device, max_diff_length=100, grad_clip=None):
    trace, label = unpack_batch(batch, device)
    trace = trace.permute(2, 0, 1)
    model.train()
    diff_length = np.random.randint(1, max_diff_length+1)
    #start_diff = np.random.randint(0, len(trace)-max_diff_length)
    #end_diff = start_diff + diff_length
    start_diff = 0
    end_diff = 700
    mask, pred = model.forward(trace, start_diff=start_diff, end_diff=end_diff)
    mask_loss = mask_loss_fn(mask, -trace)
    pred_loss = pred_loss_fn(pred, label)
    optimizer.zero_grad()
    mask_loss.backward()#retain_graph=True)
    #pred_loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip, norm_type=2)
    #optimizer.step()
    return {'mask_loss': to_np(mask_loss),
            'pred_loss': to_np(pred_loss)}

@torch.no_grad()
def eval_step(batch, model, mask_loss_fn, pred_loss_fn, device):
    trace, label = unpack_batch(batch, device)
    trace = trace.permute(2, 0, 1)
    model.eval()
    mask, pred = model.forward(trace, start_diff=len(trace))
    mask_loss = mask_loss_fn(mask, -trace)
    pred_loss = pred_loss_fn(pred, label)
    return {'mask_loss': to_np(mask_loss),
            'pred_loss': to_np(pred_loss)}