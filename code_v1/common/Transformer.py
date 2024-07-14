""" 
    Function:   Transformer 
    Refs:   Attention is All you Need
            https://github.com/SamLynnEvans/Transformer
"""
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from timm.models.layers import trunc_normal_

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, nhead, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.h = nhead
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)  # FC processes the last dimension


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, max_seq_len=1000, dropout=0.1, trainable=False):
        super(PositionalEncoder, self).__init__()
        self.trainable = trainable
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        if self.trainable:
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
            trunc_normal_(self.pos_embedding)
        else:
            # create constant 'pe' matrix with values dependant on pos and i
            pe = torch.zeros(max_seq_len, d_model)
            for pos in range(max_seq_len):
                for i in range(0, d_model, 2):
                    pe[pos, i] = math.sin(pos / (10000**((2 * i) / d_model)))
                    pe[pos, i + 1] = math.cos(pos / (10000**((2 * (i + 1)) / d_model)))
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

    def forward(self, x):
        nbatch = x.shape[0]
        if self.trainable:
            if x.is_cuda:
                self.pos_embedding.cuda()
            pos_embedding = self.pos_embedding.expand(nbatch, -1, -1)
            x = x + pos_embedding
        else:
            # make embeddings relatively larger
            x = x * math.sqrt(self.d_model)
            # add constant to embedding
            seq_len = x.size(1)
            pe = Variable(self.pe[:, :seq_len], requires_grad=False)
            pe = pe.expand(nbatch, -1, -1)
            if x.is_cuda:
                pe.cuda()  # ???check
            x = x + pe.cuda()

        return self.dropout(x)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, nhead, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(nhead, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class TransformerEncoder(nn.Module):

    def __init__(self, d_model, d_ff, nlayer, nhead, dropout):
        super(TransformerEncoder, self).__init__()
        self.N = nlayer
        self.pos_embedding = PositionalEncoder(d_model, dropout=dropout)
        self.encoder_layers = clones(EncoderLayer(d_model, d_ff, nhead, dropout), self.N)
        self.norm = LayerNorm(d_model)

    def forward(self, x, add_same_one, mask=None):
        x = self.pos_embedding(x)
        for i in range(self.N):
            x = self.encoder_layers[i](x, mask)

            if add_same_one:
                x_one = torch.mean(x, dim=1)
                x_one = x_one[:, np.newaxis, :].expand(x.shape[0], x.shape[1], x.shape[-1])
                x = x + x_one

        return self.norm(x)


# # build a decoder layer with two multi-head attention layers and one feed-forward layer
# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, heads, dropout=0.1):
#         super(DecoderLayer, self).__init__()
#         self.norm_1 = LayerNorm(d_model)
#         self.norm_2 = LayerNorm(d_model)
#         self.norm_3 = LayerNorm(d_model)

#         self.dropout_1 = nn.Dropout(dropout)
#         self.dropout_2 = nn.Dropout(dropout)
#         self.dropout_3 = nn.Dropout(dropout)

#         self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
#         self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
#         self.ff = FeedForward(d_model, dropout=dropout)

#     def forward(self, x, e_outputs, src_mask, trg_mask):
#         x2 = self.norm_1(x)
#         x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
#         x2 = self.norm_2(x)
#         x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
#         x2 = self.norm_3(x)
#         x = x + self.dropout_3(self.ff(x2))
#         return x

# class TransformerDecoder(nn.Module):
#     def __init__(self, d_model, N, heads, dropout):
#         super(TransformerDecoder, self).__init__()
#         self.N = N
#         self.pos_embedding = PositionalEncoder(d_model, dropout=dropout)
#         self.layers = clones(DecoderLayer(d_model, heads, dropout), N)
#         self.norm = LayerNorm(d_model)
#     def forward(self, trg, e_outputs, src_mask, trg_mask):
#         x = self.pos_embedding(x)
#         for i in range(self.N):
#             x = self.layers[i](x, e_outputs, src_mask, trg_mask)
#         return self.norm(x)
