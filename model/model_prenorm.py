import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()  # 调用父类的构造方法
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # pos：[max_len,1]
        # 先把括号内的分式求出来,pos是[maxlen,1],分母是[d_model],通过广播机制相乘后是[maxlen,d_model]
        div_term = pos / pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        # 再取正余弦
        pe[:, 0::2] = torch.sin(div_term)
        pe[:, 1::2] = torch.cos(div_term)

        pe = pe.unsqueeze(0)  # 增加batch维度
        self.register_buffer("pe", pe)  # 将pe矩阵进行注册为一个缓存区，自动成为模型中的参数，但是不会随着梯度进行更新

    def forward(self, x):
        x = x * math.sqrt(self.d_model)  # 放缩x
        x = x + self.pe[:, :x.size(1)]  # 直接将位置编码和x进行相加
        return x


class MutilHeadAttetion(nn.Module):
    def __init__(self, d_model, heads, dropout_prob=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads  # 定义每头最后一维的维度
        self.heads = heads  # 定义注意力头数
        self.d_model = d_model  # 定义最后一维维度
        self.w_q = nn.Linear(d_model, d_model)  # wq * x = q
        self.w_k = nn.Linear(d_model, d_model)  # wk * x = k
        self.w_v = nn.Linear(d_model, d_model)  # wv * x = v
        self.dropout = nn.Dropout(p=dropout_prob)
        self.out = nn.Linear(d_model, d_model)  # 合并每个头的输出

    # 计算 “Scaled Dot-Product Attention”
    def attetion(self, q, k, v, d_k, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill_(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask):
        bs = q.size(0)
        # 将qkv切分成多头
        q = self.w_q(q).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(bs, -1, self.heads, self.d_k).transpose(1, 2)
        scores = self.attetion(q, k, v, self.d_k, mask)
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)  # 合并多头输出
        output = self.out(concat)
        return output


class PositionwiseFeedForword(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout_prob=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps  # 防止除零错

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, heads, dropout_prob=0.1):
        super().__init__()
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.norm_3 = LayerNorm(d_model)
        self.attn = MutilHeadAttetion(d_model, heads, dropout_prob)
        self.ffn = PositionwiseFeedForword(d_model, d_ffn)
        self.dropout_1 = nn.Dropout(dropout_prob)
        self.dropout_2 = nn.Dropout(dropout_prob)

    def forward(self, enc_input, mask):
        enc_input = self.norm_1(enc_input)
        enc_ouput = enc_input + self.dropout_1(self.attn(enc_input, enc_input, enc_input, mask))
        # print(enc_ouput.shape)
        enc_ouput = self.norm_2(enc_ouput)
        enc_ouput = enc_ouput + self.dropout_2(self.ffn(enc_ouput))
        # print(enc_ouput.shape)
        return self.norm_3(enc_ouput)


class Encoder(nn.Module):
    def __init__(self, enc_vocab_size, d_model, d_ffn, N, heads, dropout_prob):
        super().__init__()
        # nn.Embedding 本质是初始化一个[vocab_size x d_model]的权重矩阵，这样就把每个token都表示为一个1 x d_model的向量
        self.embed = nn.Embedding(enc_vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ffn, heads, dropout_prob) for _ in range(N)])  # 堆叠N个EncoderLayer块

    def forward(self, enc_input, enc_mask):
        '''enc_input: [batch_size, src_len]'''
        enc_out = self.embed(enc_input)
        enc_out = self.pe(enc_out)
        for layer in self.layers:
            enc_out = layer(enc_out, enc_mask)
        return enc_out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ffn, heads, dropout_prob=0.1):
        super().__init__()
        self.attn = MutilHeadAttetion(d_model, heads, dropout_prob)
        self.norm_1 = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout_prob)
        self.cross_attn = MutilHeadAttetion(d_model, heads, dropout_prob)
        self.norm_2 = LayerNorm(d_model)
        self.dropout_2 = nn.Dropout(dropout_prob)
        self.ffn = PositionwiseFeedForword(d_model, d_ffn)
        self.norm_3 = LayerNorm(d_model)
        self.dropout_3 = nn.Dropout(dropout_prob)
        self.norm_4 = LayerNorm(d_model)

    def forward(self, enc_out, enc_mask, dec_input, dec_mask):
        dec_input = self.norm_1(dec_input)
        dec_out = dec_input + self.dropout_1(self.attn(dec_input, dec_input, dec_input, dec_mask))

        residual = dec_out
        dec_out = self.norm_2(dec_out)
        dec_out = residual + self.dropout_2(self.cross_attn(dec_out, enc_out, enc_out, enc_mask))

        residual = dec_out
        dec_out = residual + self.dropout_3(self.ffn(dec_out))
        return self.norm_4(dec_out)


class Decoder(nn.Module):
    def __init__(self, dec_vocab_size, d_model, d_ffn, N, heads, dropout_prob):
        super().__init__()
        self.embed = nn.Embedding(dec_vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, d_ffn, heads, dropout_prob) for _ in range(N)])

    def forward(self, enc_out, enc_mask, dec_input, dec_mask):
        dec_out = self.embed(dec_input)
        dec_out = self.pe(dec_out)
        for layer in self.layers:
            dec_out = layer(enc_out, enc_mask, dec_out, dec_mask)
        return dec_out

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class Transformer(nn.Module):
    '''
    src_vocab_size: 源词典的大小
    tgt_vocab_size: 目标词典的大小
    d_model: 每个torch表示的向量长度
    d_ffn: feedforward中隐藏层的神经元个数
    N: transformer block堆叠的个数
    heads: 注意力切分的头数
    dropout_prob: 丢弃概率
    '''

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, d_ffn, N, heads, dropout_prob):
        super().__init__()
        self.encode = Encoder(src_vocab_size, d_model, d_ffn, N, heads, dropout_prob)
        self.decode = Decoder(tgt_vocab_size, d_model, d_ffn, N, heads, dropout_prob)
        self.project = nn.Linear(d_model, tgt_vocab_size)
        # self.apply(init_weights)

    def forward(self, enc_input, dec_input, enc_mask, dec_mask):
        '''
        enc_input: [batch_size, src_len]
        dec_input: [batch_size, tgt_len]
        '''
        enc_out = self.encode(enc_input, enc_mask)
        dec_out = self.decode(enc_out, enc_mask, dec_input, dec_mask)
        project_out = self.project(dec_out)
        return project_out  # [batch_size , tgt_len, tgt_vocab_size]
