'''
    本实现主要参考:https://zhuanlan.zhihu.com/p/347709112
'''
from torch import nn
import torch
import torch.nn.functional as F
import math
from einops import rearrange
import random
from torch.nn.modules.module import Module

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len, poem_type):
        super().__init__()
        self.d_model = d_model

        # 正余弦绝对位置编码，根据pos和i创建一个常量pe矩阵
        # 写法参考https://blog.csdn.net/Flying_sfeng/article/details/100996524
        if poem_type=='sinusoidal':
            pe = torch.zeros(max_seq_len, d_model)
            pos = torch.arange(0, max_seq_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div_term)
            pe[:, 1::2] = torch.cos(pos * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        
        # 可学习绝对位置编码
        elif poem_type=='learnable':
            self.pe = nn.Parameter(torch.randn(1, max_seq_len, d_model))

    def forward(self, x):
        # 让embeddings vector 相对大一些
        x = x * math.sqrt(self.d_model)
        # 增加位置常量
        seq_len = x.size(1)
        x = (x + self.pe[:, :seq_len]).to(x.device)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    # q.shape:[bs,N,sl,d_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # mask掉那些为了padding长度增加的token，让其通过softmax计算后为0
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(d_model, d_model),
                                 nn.Dropout(dropout))


    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        # get dimensions bs * sl * N * d_k
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)  # the same dimension as input k:[bs,sl,d_model]
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, activate):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.bn1 = nn.BatchNorm1d(d_ff)
        self.dropout_1 = nn.Dropout(dropout)

        # self.conv = nn.Conv2d(in_channels=d_ff,out_channels=d_ff,kernel_size=3,padding=1,groups=d_ff)
        # self.bn2 = nn.BatchNorm2d(d_ff)
        # self.dropout_2 = nn.Dropout(dropout)

        self.linear_2 = nn.Linear(d_ff, d_model)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.dropout_3 = nn.Dropout(dropout)

        self.activate = activate

    def Activate(self, x):
        if self.activate == 'relu':
            x = F.relu(x)
        elif self.activate == 'gelu':
            x = F.gelu(x)
        return x

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.linear_1(x)
        x = rearrange(x,'b l d -> b d l')
        x = self.bn1(x)
        x = rearrange(x, 'b d l -> b l d')
        x = self.Activate(x)
        x = self.dropout_1(x)
        
        # x = rearrange(x, 'b (h w) d -> b d h w', h=int(math.sqrt(seq_len)), w=int(math.sqrt(seq_len)))
        # x = self.conv(x)
        # x = self.bn2(x)
        # x = rearrange(x, 'b d h w -> b (h w) d')
        # x = self.Activate(x)
        # x = self.dropout_2(x)
        
        x = self.linear_2(x)
        x = rearrange(x, 'b l d -> b d l')
        x = self.bn3(x)
        x = rearrange(x, 'b d l -> b l d')
        x = self.dropout_3(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, activate):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=d_ff, dropout=dropout,activate=activate)

    def forward(self, x_q, x_k=None, x_v=None, mask=None):
        q = self.norm_1(x_q)
        if x_k is None and x_v is None:
            k, v = q, q
        else:
            k = self.norm_1(x_k)
            v = self.norm_1(x_v)
        x = x_q + self.attn(q, k, v, mask)
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x


class TrmEncoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout, d_ff, activate='gelu'):
        super().__init__()
        self.N = N
        self.layers = nn.ModuleList([])
        for _ in range(N):
            self.layers.append(EncoderLayer(d_model, heads, d_ff, dropout, activate))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k=None, v=None, mask=None):
        '''

        Args:
            q: shape:[bs,seq_len,d_model]
            v: shape:[bs,seq_len,d_model]
            k: shape:[bs,seq_len,d_model]
            mask:

        Returns:

        '''
        if k is None or v is None:
            for i in range(self.N):
                q = self.layers[i](q, mask)
        else:
            for i in range(self.N):
                q = self.layers[i](q, k, v, mask)
        return self.norm(q)


class SequenceCutOut(Module):
    def __init__(self, prob, max_num):
        super().__init__()
        self.prob = prob
        self.max_num = max_num

    def forward(self, x):
        # x:[b,L,C]
        if random.random() < self.prob:
            if isinstance(x, tuple):
                seq_len = x[0].shape[1]
            else:
                seq_len = x.shape[1]
            cutout_num = random.randint(1, self.max_num)
            cutout_index = list(range(seq_len))
            cutout_index.remove(seq_len//2)
            cutout_index = random.sample(cutout_index, cutout_num)
            tensor_index = (torch.arange(seq_len)+1).bool()
            tensor_index[cutout_index]=False
            if isinstance(x, tuple):
                return x[0][:,tensor_index,:], x[1][:,tensor_index,:]
            else:
                return x[:,tensor_index,:]
        else:
            return x