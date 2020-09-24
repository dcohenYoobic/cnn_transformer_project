import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#Calculates self_attention for sublayers
class attention(nn.Module):

    def __init__(self, temp, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.temp = temp

    def forward(self, q, k, v, mask=None):

        #Calculates Q*K_T
        attention = torch.matmul(q / self.temp, k.transpose(2,3))

        #Apply masking operation
        if mask is not None:
        
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = self.dropout(F.softmax(attention, dim = -1))
        out = torch.matmul(attention, v)
        return out, attention

            
class MultiHeadAttention(nn.Module):

    def __init__(self, heads, d_model, d_k, d_v, dropout = 0.1):
        super().__init__()

        self.heads = heads
        self.d_k = d_k
        self.d_v = d_v

        self.q_linear = nn.Linear(d_model, heads * d_k, bias = False)
        self.k_linear = nn.Linear(d_model, heads * d_k, bias = False)
        self.v_linear = nn.Linear(d_model, heads * d_v, bias = False)
        self.o_linear = nn.Linear(heads * d_v, d_model, bias = False)

        self.attn = attention(temp = np.sqrt(d_k))
        self.dropout = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm(d_model, eps =1e-6)
        
    def forward(self, q, k, v, mask = None):

        bs, qs = q.size(0), q.size(1)
        res = q
        #Linear operations, split into h heads, and transpose for attention dot product to get dimension : bs * n * seq_len * d_model
        q = (self.q_linear(q).view(bs, qs, self.heads, self.d_k)).transpose(1,2)
        k = (self.k_linear(k).view(bs, k.size(1), self.heads, self.d_k)).transpose(1,2)
        v = (self.v_linear(v).view(bs, v.size(1), self.heads, self.d_v)).transpose(1,2)
        
        #q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attention = self.attn(q, k, v, mask = mask)
        
        # Transpose to get dimension: b * lq * n * dv
        # Concatenate all the heads together: bs * lq * n * dv

        q = q.transpose(1,2).contiguous().view(bs, qs, -1)
        
        q = self.dropout(self.o_linear(q)) + res
        q = self.layerNorm(q)
        return q, attention

#Position-Wise FeedèForward Networks consists of two linear transformations with a ReLU activation between.
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm(d_model, eps = 1e-6)
    
    def forward(self, inputs):

        #out = self.linear_2(F.relu(self.linear_1(inputs)))
        #out = self.dropout(out) + inputs
        #out = self.layerNorm(out)
        
        
        residual = inputs

        inputs = self.w_2(F.relu(self.w_1(inputs)))
        inputs = self.dropout(inputs)
        inputs += residual

        inputs = self.layer_norm(inputs)

        
        return inputs





