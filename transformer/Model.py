import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import *


def subseq_mask(sequence):
    bs, lens = sequence.size()
    return (1 - torch.triu(torch.ones((1, lens, lens), device=sequence.device), diagonal=1)).bool()

def padding_mask(sequence, i):
    return (i != sequence).unsqueeze(-2)

#Calculates Positional Encoding Matrix
class PosEncoding(nn.Module):

    def __init__(self, d, n = 200):
        super(PosEncoding, self).__init__()
        #print('\n \n \n DDD', d, '\n \n')
        self.register_buffer('pos_table', self.sin_table(n, d))
        #print('\n \n \n TAILLE', self.pos_table.size(), '\n \n')
    def sin_table(self, n, d):

        def pos_angle(pos):
            return [pos/np.power(10000,2*(j//2)/d) for j in range(d)]

        pe = np.array([pos_angle(pos) for pos in range(n)])
        #pe[:, 0::2] = np.sin(pe[:, 0::2]) #for even dimension i Pe = sin(pos/10000^(2i/d_model)
        #pe[:, 1::2] = np.cos(pe[:, 1::2]) ##for odd dimension i Pe = cos(pos/10000^(2i/d_model)
        
        pe[:, 0::2] = np.sin(pe[:, 0::2])  
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        
        return torch.FloatTensor(pe).unsqueeze(0)

    def forward(self, inputs):
        return inputs + self.pos_table[:, inputs.size(1)].clone().detach()


class Encoder(nn.Module):

    def __init__(self, src_voc_size, d_embed, layers, heads, d_k, d_v, d_model, d_ff, i, n = 200, dropout = 0.1):

        super().__init__()
        self.src_embed = nn.Embedding(src_voc_size, d_embed, padding_idx = i)
        self.pos = PosEncoding(d_embed, n = n)
        self.dropout = nn.Dropout(p = dropout)
        self.layers_stack = nn.ModuleList([EncoderLayer(d_model, d_ff, heads, d_k, d_v, dropout = dropout) for _ in range(n)])
        self.layerNorm = nn.LayerNorm(d_model, eps = 1e-6)

    def forward(self, sequence, mask, return_attns = False):

        enc_attentions = []

        enc_out = self.dropout(self.pos(self.src_embed(sequence)))
        enc_out = self.layerNorm(enc_out)

        for l in self.layers_stack:
            enc_out, enc_attention = l(enc_out, f_mask = mask)
            enc_attentions += [enc_attention] if return_attns else []

        if return_attns:
            return enc_out, enc_attentions

        return enc_out

class Decoder(nn.Module):

    def __init__(self, trg_voc_size, d_embed, layers, heads, d_k, d_v, d_model, d_ff, i, n = 200, dropout = 0.1) :
        super().__init__()

        self.trg_embed = nn.Embedding(trg_voc_size, d_embed, padding_idx = i)
        self.pos = PosEncoding(d_embed, n = n)
        self.dropout = nn.Dropout(p = dropout)
        self.layers_stack = nn.ModuleList([DecoderLayer(d_model, d_ff, heads, d_k, d_v, dropout = dropout) for _ in range(layers)])
        self.layerNorm = nn.LayerNorm(d_model, eps = 1e-6)

    def forward(self, sequence, trg_mask, enc_out, src_mask, return_attns = False):

        dec_attentions, dec_enc_attentions = [], []

        dec_out = self.dropout(self.pos(self.trg_embed(sequence)))
        dec_out = self.layerNorm(dec_out)

        for l in self.layers_stack:
            dec_out, dec_attention, dec_enc_attention = l(dec_out, enc_out, mask1 = trg_mask, mask2 = src_mask)
            dec_attentions += [dec_attention] if return_attns else []
            dec_enc_attentions += [dec_enc_attention] if return_attns else []

        if return_attns:
            return dec_out, dec_attentions, dec_enc_attentions

        return dec_out
    

class Transformer(nn.Module):

    def __init__(self, src_voc_size, trg_voc_size, pad1, pad2, d_embed = 512, d_model = 512, d_ff = 2048,
                 layers = 6, heads = 8, d_k = 64, d_v = 64, dropout = 0.1, n = 200, trg_weight_sharing = True,
                 src_weight_sharing = True):
        super().__init__()
        self.pad1 = pad1
        self.pad2 = pad2

        self.encoder = Encoder(src_voc_size =src_voc_size, n = n, d_embed = d_embed, d_model=d_model, d_ff = d_ff, layers = layers, heads = heads, d_k = d_k, d_v = d_v, i = pad1)
        self.decoder = Decoder(trg_voc_size =trg_voc_size, n = n, d_embed = d_embed, d_model=d_model, d_ff = d_ff, layers = layers, heads = heads, d_k = d_k, d_v = d_v, i = pad2) 
        self.target_word = nn.Linear(d_model, trg_voc_size, bias = False)

        for param in self.parameters():
            if param.dim()>1:
                nn.init.xavier_uniform_(param)

        assert d_model == d_embed
        self.x_logit_scale = 1

        if trg_weight_sharing:
            self.target_word = self.decoder.trg_embed.weight
            self.x_logit_scale = d_model ** (-0.5)

        if src_weight_sharing:
            self.encoder.src_embed.weight = self.decoder.trg_embed.weight

    def forward(self, src, trg):

        mask1 = padding_mask(src, self.pad1)
        mask2 = padding_mask(trg, self.pad2) & subseq_mask(trg)
        enc_out, *_ = self.encoder(src, mask1)
        dec_out, *_ = self.decoder(trg, mask2, enc_out, mask1)
        logits = self.trg_embed(dec_out)*self.x_logit_scale

        return logits.view(-1, logits.size(2))
