# Translation of the following from Tensorflow to Pytorch:
#  https://github.com/suvadeep-iitb/TransNet/blob/master/transformer.py

from copy import deepcopy
import numpy as np
import torch
from torch import nn

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()
        self.inv_freq = 1/(1e5**(torch.range(0, demb, 2.0)/demb))
    
    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.einsum('i,j->ij', pos_seq, self.inv_freq)
        pos_emb = torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), -1)
        if bsz is not None:
            return torch.tile(pod_rmb[:, None, :], [1, bsz, 1])
        else:
            return pos_emb[:, None, :]

class PositionwiseFF(nn.Module):
    def __init__(self, input_shape, d_model, d_inner, dropout, kernel_initializer):
        super().__init__()
        
        self.input_shape = input_shape
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        
        self.layer_1 = nn.Linear(in_features=1, out_features=self.d_inner) ## FIXME
        self.kernel_initializer(self.layer_1.weight)
        self.relu_1 = nn.ReLU()
        self.drop_1 = nn.Dropout(self.dropout)
        self.layer_2 = nn.Linear(in_features=1, out_features=self.d_model) ## FIXME
        self.kernel_initializer(self.layer_2.weight)
        self.drop_2 = nn.Dropout(self.dropout)
        
    def forward(self, x):
        core_out = self.layer_1(x)
        core_out = self.relu_1(core_out)
        core_out = self.drop_1(core_out)
        core_out = self.layer_2(core_out)
        core_out = self.drop_2(core_out)
        output = [core_out + inputs]
        return output

class RelativeMultiHeadAttn(nn.Module):
    def __init__(
        self,
        input_shape,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt,
        kernel_initializer,
        r_r_bias=None,
        r_w_bias=None,
        smooth_pos_emb=True,
        untie_pos_emb=True,
        clamp_len=-1):
        super().__init__()
        
        self.input_shape = input_shape
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.dropatt = dropatt
        self.kernel_initializer = kernel_initializer
        self.smooth_pos_emb = smooth_pos_emb
        self.untie_pos_emb = untie_pos_emb
        self.clamp_len = clamp_len
        
        self.qkv_net = nn.Linear(in_features=input_shape[1], out_features=3*self.n_head*self.d_head, bias=False)
        self.kernel_initializer(self.qkv_net.weight)
        
        if self.smooth_pos_emb:
            self.r_net = nn.Linear(in_features=1, out_features=self.n_head*self.d_head, bias=False) ## FIXME
            self.kernel_initializer(self.r_net.weight)
        elif self.untie_pos_emb:
            self.pos_emb = nn.Embedding(num_embeddings=2*self.clamp_len+1, embedding_dim=self.d_model)
        
        self.drop_r = nn.Dropout(self.dropout)
        self.drop = nn.Dropout(self.dropout)
        self.dropatt = nn.Dropout(self.dropatt)
        self.o_net = nn.Linear(in_features=1, out_features=self.d_model, bias=False) ## FIXME
        self.kernel_initializer(self.o_net.weight)
        self.scale = 1/(self.d_head**2)
        
        if r_r_bias is not None and r_w_bias is not None:
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        else:
            self.r_r_bias = nn.Parameter(data=torch.zeros(self.n_head, self.d_head), requires_grad=True)
            self.r_w_bias = nn.Parameter(data=torch.zeros(self.n_head, self.d_head), requires_grad=True)
    
    def _rel_shift(self, x):
        x_shape = deepcopy(x.shape)
        x = nn.functional.pad(x, ((0, 0), (1, 0), (0, 0), (0, 0)))
        x = torch.reshape(x, (x_shape[1]+1, x_shape[0], x_shape[2], x_shape[3]))
        x = torch.narrow(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = torch.reshape(x, x_shape)
        return x
    
    def forward(self, inputs):
        w, r = inputs
        qlen, rlen, bsz = w.shape[0], r.shape[0], w.shape[1]
        w_heads = self.qkv_net(w)
        if not self.smooth_pos_emb and self.untie_pos_emb:
            r = self.pos_emb(r)
        r_drop = self.drop_r(r)
        if self.smooth_pos_emb:
            r_head_k = self.r_net(r_drop)
        else:
            r_head_k = r_drop
        w_head_q, w_head_k, w_head_v = torch.split(w_heads, 3, dim=-1)
        w_head_q = w_head_q[-qlen:]
        klen = w_head_k.shape[0]
        w_head_q = torch.reshape(w_head_q, (qlen, bsz, self.n_head, self.d_head))
        w_head_k = torch.reshape(w_head_k, (klen, bsz, self.n_head, self.d_head))
        w_head_v = torch.reshape(w_head_v, (klen, bsz, self.n_head, self.d_head))
        r_head_k = torch.reshape(r_head_k, (rlen, self.n_head, self.d_head))
        rw_head_q = w_head_q + self.r_w_bias
        rr_head_q = w_head_q + self.r_r_bias
        AC = torch.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)
        BD = torch.einsum('ibnd,jnd->ijbn', rr_head_q, r_head_k)
        BD = self._rel_shift(BD)
        BD = BD[:, :klen, :, :]
        attn_score = AC+BD
        attn_score = attn_score * self.scale
        attn_prob = nn.functional.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        attn_vec = torch.reshape(attn_vec, (attn_vec.shape[0], attn_vec.shape[1], self.n_head*self.d_head))
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)
        outputs = [w+attn_out, attn_prob, AC, BD]
        return outputs

class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_shape,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        initializer,
        r_w_bias=None,
        r_r_bias=None,
        smooth_pos_emb=True,
        untie_pos_emb=True,
        clamp_len=-1):
        super().__init__()
        
        self.input_shape = input_shape
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.d_inner = d_inner
        self.dropout = dropout
        self.dropatt = dropatt
        self.initializer = initializer
        self.smooth_pos_emb = smooth_pos_emb
        self.untie_pos_emb = untie_pos_emb
        self.clamp_len = clamp_len
        
        self.xltran_attn = RelativeMultiHeadAttn(
            input_shape=self.input_shape,
            n_head=self.n_head,
            d_model=self.d_model,
            d_head=self.d_head,
            dropout=self.dropout,
            dropatt=self.dropatt,
            kernel_initializer=self.initializer,
            r_w_bias=r_w_bias,
            r_r_bias=r_r_bias,
            smooth_pos_emb=self.smooth_pos_emb,
            untie_pos_emb=self.untie_pos_emb,
            clamp_len=self.clamp_len)
        
        self.pos_ff = PositionwiseFF(
            input_shape=self.input_shape,
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout,
            kernel_initializer=self.initializer)
        
    def forward(self, x):
        inp, r = x
        attn_outputs = self.xltran_attn(inp, r)
        ff_output = self.pos_ff(attn_outputs[0])
        output = [ff_output[0]] + attn_outputs[1:]
        return output

class Transformer(nn.Module):
    def __init__(
        self,
        input_shape,
        n_layer, # Number of transformer layers
        d_model, # Output dimension of each transformer layer
        n_head, # Number of heads in multi-head self-attention layers
        d_head,
        d_inner,
        dropout,
        dropatt,
        n_classes,
        conv_kernel_size, # Kernel size for conv1d input embedding
        pool_size, # Factor by which to downsample the input sequence
        initializer,
        clamp_len=-1,
        untie_r=False,
        smooth_pos_emb=True,
        untie_pos_emb=True,
        output_attn=False):
        super().__init__()
        
        self.input_shape = input_shape
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.dropout = dropout
        self.dropatt = dropatt
        self.n_classes = n_classes
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.clamp_len = clamp_len
        self.untie_r = untie_r
        self.smooth_pos_emb = smooth_pos_emb
        self.untie_pos_emb = untie_pos_emb
        self.output_attn = output_attn
        self.initializer = initializer
        
        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=self.d_model, kernel_size=self.conv_kernel_size)
        self.relu_1 = nn.ReLU()
        self.pool_1 = nn.AvgPool1d(kernel_size=self.pool_size, stride=self.pool_size)
        
        if self.smooth_pos_emb:
            self.pos_emb = PositionalEmbedding(self.d_model)
        else:
            assert self.clamp_len > 0
            if not self.untie_pos_emb:
                self.pos_emb = nn.Embedding(num_embeddings=2*self.clamp_len+1, embedding_dim=self.d_model)
            else:
                self.pos_emb = None
        
        if not self.untie_r:
            self.r_w_bias = nn.Parameter(data=torch.zeros(self.n_head, self.d_head), requires_grad=True)
            self.r_r_bias = nn.Parameter(data=torch.zeros(self.n_head, self.d_head), requires_grad=True)
        
        self.tran_layers = nn.ModuleList([
            TransformerLayer(
                input_shape=self.input_shape,
                n_head=self.n_head,
                d_model=self.d_model,
                d_head=self.d_head,
                d_inner=self.d_inner,
                dropout=self.dropout,
                dropatt=self.dropatt,
                initializer=self.initializer,
                r_w_bias=None if self.untie_r else self.r_w_bias,
                r_r_bias=None if self.untie_r else self.r_r_bias,
                smooth_pos_emb=self.smooth_pos_emb,
                untie_pos_emb=self.untie_pos_emb,
                clamp_len=self.clamp_len) for _ in range(self.n_layer)])
        
        self.out_dropout = nn.Dropout(self.dropout)
        self.fc_output = nn.Linear(in_features=1, out_features=self.n_classes) ## FIXME
        
    def forward(self, inp):
        print(inp.shape)
        inp = inp.unsqueeze(dim=-1)
        print(inp.shape)
        inp = self.conv_1(inp)
        print(inp.shape)
        inp = self.relu_1(inp)
        print(inp.shape)
        inp = self.pool_1(inp)
        print(inp.shape)
        inp = torch.permute(inp, (1, 0, 2))
        print(inp.shape)
        slen = inp.shape[0]
        pos_seq = torch.range(slen-1, -slen, -1.0)
        if self.clamp_len > 0:
            pos_seq = torch.minimum(pos_seq, self.clamp_len)
            pos_seq = torch.maximum(pos_seq, -self.clamp_len)
        if self.smooth_pos_emb:
            pos_emb = self.pos_emb(pos_seq)
        else:
            pos_seq = pos_seq + torch.abs(torch.min(pos_seq))
            pos_emb = pos_seq if self.untie_pos_emb else self.pos_emb(pos_seq)
        core_out = inp
        out_list = []
        for i, layer in enumerate(self.tran_layers):
            all_out = layer([core_out, pos_emb])
            core_out = all_out[0]
            print(core_out.shape)
            out_list.append(all_out[1:])
        core_out = self.out_dropout(core_out)
        print(core_out.shape)
        output = torch.mean(core_out, dim=0)
        print(core_out.shape)
        scores = self.fc_output(output)
        print(scores.shape)
        if self.output_attn:
            for i in range(len(out_list)):
                for j in range(len(out_list[i])):
                    out_list[i][j] = torch.permute(out_list[i][j], (2, 3, 0, 1))
            return [scores] + out_list
        else:
            return [scores]
        