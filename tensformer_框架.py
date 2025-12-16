import torch
import torch.nn as nn
import math


# 自注意力
class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # 对10%神经元做一个随机失活防止过拟合
        self.softmax = nn.Softmax(dim=-1)  # 将得分准换为概率分布，在最后一个维度

    def forward(self, Q, K, V, mask=None):
        """
        X：batch  seq_len  d_model
        batch：一次送到模型的句子个数，seq_len：句子中token数量；d_model:embeding维度
        Q:query向量，维度：batch，heads , seq_len_q , d_k
        K:key 向量，维度：batch，heads , seq_len_k , d_k
        V:value 向量，维度：batch，heads , seq_len_v , d_v
        mask:告诉模型哪些位置需要忽略
        """

        d_k = Q.size(-1)  # Q最后一维d_k,对query缩放
        # batch，heads , seq_len_q , d_k乘上batch，heads , d_k, seq_len_k ->batch，heads ,seq_len_q , seq_len_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # 进行缩放，梯度更稳定
        # 如果提供mask，通过mask==0来找到需要屏蔽位置，将这些值改为负无穷
        # softmax之后这些位置会变成0
        # mask==0（被屏蔽）==1（位置可见）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        # batch，heads ,seq_len_q , seq_len_k,对最后一维进行softmax，对key进行，得query的key到权重
        attn = self.softmax(scores)
        attn = self.dropout(attn)  # 防止过拟合
        out = torch.matmul(attn, V)  # 结果 batch，heads，seq_len_q，d_k
        return out, attn


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        '''
        d_model:embeding维度 512
        n_head:多头注意力头数 8
        d_model 需要被n_head整除 64
        '''
        assert d_model % n_head == 0
        self.d_k = d_model // n_head  # 每个头维度
        self.n_head = n_head

        # 将输入映射到Q K V，通过线下映射让模型有学习能力
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)  # 模型融合不同头信息，多头拼接后映射回d_model

        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)  # 防止过拟合
        self.norm = nn.LayerNorm(d_model)  # 对残差后的归一化

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        # Q为 batch , self.n_head , seq_len , self.d_k
        # 让每个注意力独立处理整个序列，方便后续计算注意力权重
        Q = self.W_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 计算注意力
        out, attn = self.attention(Q, K, V, mask)
        # out: betch , seq_len , d_model
        # 多头拼接
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_k)
        out = self.fc(out)  # 让输入和输出一致，方便残差链接
        out = self.dropout(out)  # 在训练阶段丢弃一部分神经元，防止过拟合
        # out+q残差连接,layernorm
        return self.norm(out + q), attn  # 返回输出和注意力权重


# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 输入维度为d_model,输出为d_ff,512->2048让模型学到更丰富特征
        self.fc1 = nn.Linear(d_model, d_ff)
        # 保证第二个线性层输出维度等于第一个线性层的输入维度，方便后续残差连接
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)  # 丢弃防止过拟合，随机丢弃（失活）
        self.norm = nn.LayerNorm(d_model)  # 对最后一维归一化

    def forward(self, x):
        """
        对x进行fc1，得到输出
        经过激活函数relu
        防止过拟合
        经过第二个线性层
        return: out+x（残差连接）,再层归一化
        X形状: batch , seq_len , de_model
        """
        out = self.fc2(self.dropout(torch.relu(self.fc1(x))))
        return self.norm(out + x)  # 残差连接目的：保留输入低价信息，避免训练时候信息丢失


# 编码层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 多头注意力机制,输入维src实现序列内部信息交互，每个token可以看到序列中其它token，上下文依赖
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 对每个位置向量独立进行非线性变化,提升模型表达能力
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, src, src_mask=None):
        # src 输入序列张量：batch,seq_len,d_model
        # src_mask 屏蔽padding位置，避免模型管制无效token(encoder),decoder中mask用来防止看到未来词
        out, _ = self.self_attn(src, src, src, src_mask)
        # 通过前馈神经网络，每个位置token通过单独两层线性层映射和激活函数，提升表达能力
        out = self.ffn(out)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Mask多头注意力机制
        # 输入tgt(目标序列)
        # 计算目标序列内部的自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 交叉注意力，和encoder交互
        # 输入Q：当前解码器的输出，K=V=编码器的memory
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)  # 提升表达能力

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        tgt:目标序列 memory：编码器输出
        tgt_mask：屏蔽未来token
        memory_mask:Padding做掩码
        """
        # 目标序列内部的自注意力
        out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        # 将目标序列和原来序列进行交互
        # Q：解码器输出out；K=V=memory（编码器输出）
        out, _ = self.cross_attn(out, memory,memory, memory_mask)
        out = self.ffn(out)
        return out


# 定义位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # d_model 每个词维度；max_len：句子最大长度
        # 初始化位置编码矩阵 形状为max_len,d_model
        pe = torch.zeros(max_len, d_model)

        # 定义记录每个token位置索引，0-max_len-1
        # [max_len,1] 方便后续与缩放因子进行相处
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 每个维度的放缩因子；2i
        # orch.arange(0,d_model,2)生成偶数维度的索引
        # (2i/d_model)*(-ln(10000.0))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 每个token的位置索引 * 每个维度缩放因子，再套上sin得到偶数维度的位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 每个token的位置索引 * 每个维度缩放因子，再套上cos得到偶数维度的位置编码
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加batch维度，1,max_len,d_model,方便后续与输入embedding进行相加
        pe = pe.unsqueeze(0)
        # 注册为buffer，把位置编码pe存在模型，不参与训练，但随着模型保存/加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:输出的embedding 形状：batch seq_len d_model
        seq_len = x.size(1)
        '''
        每个token的embedding加上对应位置的编码
        self.pe[:, :seq_len, :]取前seq_len个位置
        x + self.pe[:, :seq_len, :]形状为：batch,seq_len,d_model
        transformer 就可以知道token的顺序
        '''
        return x + self.pe[:, :seq_len, :]


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layer, d_ff, dropout=0.1, max_len=5000):
        super().__init__()
        # 词嵌入层：vocab_size:词表大小，包含不同taken总数
        # 将token id（对原始文本分词得到词表，不同词对应不同ID）转换为连续向量，维度为d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码加入序列中token的位置信息
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # 构建编码器的堆叠结构
        # 堆叠num_layers个encoder
        # nn.ModuleList为网络准备的列表，用来存放多个子模块
        # 列表推导式用来生成num_layers个encoder
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layer)
        ])

    def forward(self, src, src_mask=None):
        # 将输入token ID转换成embedding向量
        # 输出 shape: batch seq_len d_model
        # 乘上 sqpt(d_model) 让注意力计算更稳定
        out = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        # 位置编码
        out = self.pos_encoding(out)
        # 逐层经过encoderlayer
        for layer in self.layers:
            out = layer(out, src_mask)  # self_attn + ffn

        return out  # 返回编码输出 batch seq_len d_model


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layer, d_ff, dropout=0.1, max_len=5000):
        super().__init__()

        # 将目标序列token id 转换为向量维度为d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        # 定义解码器列表
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layer)
        ])
        # 输出投影层 将decoder的输出映射回原词汇表大小，从而得到每个token的预测分布
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
         tgt 目标序列解码器的输入
         memory编码器输出
         tgt_mask 目标序列mask
         memory_mask 屏蔽pad
        """
        out = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)  # 缩放
        # 添加位置编码
        out = self.pos_encoding(out)
        # 逐层经过decoderlayer
        for layer in self.layers:
            out = layer(out, memory, tgt_mask, memory_mask)

        # 将解码器最后一层输出隐藏向量映射回原词汇表维度，得到每个token预测向量
        return self.fc_out(out)


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab,  # 原语言词表大小
                 tgt_vocab,  # 目标语言大小
                 d_model=512,  # embedding 向量维度
                 n_heads=8,
                 num_encoder_layer=6,
                 num_decoder_layer=6,
                 d_ff=2048,  # ffn隐藏层层数
                 dropout=0.1,
                 max_len=5000
                 ):
        super().__init__()
        # 编码器，将原语言部分，编码为上下文表示
        self.encoder = Encoder(
            src_vocab, d_model, n_heads, num_encoder_layer, d_ff, dropout, max_len
        )
        # 解码器 根据编码器输出和目标语言输入生成预测
        self.decoder = Decoder(
            tgt_vocab, d_model, n_heads, num_decoder_layer, d_ff, dropout, max_len
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        memory = self.encoder(src, src_mask)
        # 解码器前向传播
        out = self.decoder(tgt, memory, tgt_mask, memory_mask)
        # 返回transformer输出 batch seq_len_tgt tgt_vocab
        return out


def generate_mask(size):
    # 生成上三角，不含对角线
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    # 明确生成上三角（需要屏蔽位置）
    return mask == 0


src_vocab = 10000  # 源语言词表大小
tgt_vocab = 10000  # 目标语言词表大小
model = Transformer(src_vocab, tgt_vocab)
src = torch.randint(0, src_vocab, (32, 10))  # 原序列batch=32 src_len=10 每个元素为tgt ID
tgt = torch.randint(0, tgt_vocab, (32, 20))

tgt_mask = generate_mask(tgt.size(1)).to(tgt.device)  # 取目标序列长度
out = model(src, tgt, tgt_mask=tgt_mask)  # 前向传播
# 每个目标token对应概率
print(out.shape)  # batch tgt_len tgt_vocab


#最终返回torch.Size([32, 20, 10000])
# 模型为 32 个样本的每个目标序列（20 个 Token），输出了每个 Token 位置在 10000 词表上的预测分布
