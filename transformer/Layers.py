import torch.nn as nn
import torch
from transformer.Sublayers import *

#Encoder Layer has two sublayers :  multi-head self-attention mechanism, and positionwise fully connected feed-forward network.
class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, heads, d_k, d_v, dropout = 0.1):

        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(heads, d_model, d_k, d_v, dropout = dropout)
        self.ff = FeedForward(d_model, d_ff, dropout = dropout)

    def forward(self, inputs, f_mask = None):
        out, attention = self.attn(inputs, inputs, inputs, mask = f_mask)
        out = self.ff(out)
        return out, attention

#Decoder Layer has three sublayers :  multi-head self-attention mechanism, encoder-decoder attention,  positionwise fully connected feed-forward network.
class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, heads, d_k, d_v, dropout = 0.1):
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadAttention(heads, d_model, d_k, d_v, dropout=dropout)
        self.enc_dec = MultiHeadAttention(heads, d_model, d_k, d_v, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout = dropout)

    def forward(self, inputs, enc_out, mask1 = None, mask2 = None):
        out, attention1 = self.attn(inputs, inputs, inputs, mask = mask1)
        out, attention2 = self.enc_dec(out, enc_out, enc_out, mask = mask2)
        out = self.ff(out)

        return out, attention1, attention2

    
