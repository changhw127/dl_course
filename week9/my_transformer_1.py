import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码模块，通过正弦/余弦函数为输入序列添加位置信息"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 嵌入维度（必须与词嵌入维度相同）
            max_len: 支持的最大序列长度
            dropout: 随机失活比例
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算频率项（用于正弦/余弦函数）
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 填充位置编码矩阵
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引使用正弦
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引使用余弦
        pe = pe.unsqueeze(0)  # 增加批次维度 [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 注册为不需要梯度的缓冲区

    def forward(self, x):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]  # 自动广播到批次维度
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, nhead, dropout=0.1):
        """
        Args:
            d_model: 输入维度
            nhead: 注意力头的数量
            dropout: 注意力权重的随机失活比例
        """
        super().__init__()
        assert d_model % nhead == 0, "d_model必须能被nhead整除"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead  # 每个头的维度

        # 定义线性变换层（Q/K/V矩阵和输出层）
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: 查询张量 [batch_size, q_len, d_model]
            k: 键张量 [batch_size, k_len, d_model]
            v: 值张量 [batch_size, v_len, d_model]
            mask: 掩码张量（用于遮挡无效位置）
        Returns:
            多头注意力输出 [batch_size, q_len, d_model]
        """
        batch_size = q.size(0)

        # 线性变换 + 分头操作
        # [batch_size, seq_len, nhead, head_dim] -> [batch_size, nhead, seq_len, head_dim]
        q = self.q_linear(q).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        # 计算缩放点积注意力 [batch_size, nhead, q_len, k_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用掩码（如有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 用极小值填充被mask的位置

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和 [batch_size, nhead, q_len, head_dim]
        context = torch.matmul(attn_weights, v)

        # 合并多头 [batch_size, q_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(context)


class FeedForward(nn.Module):
    """前馈网络（Position-wise Feed Forward Network）"""

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1):
        """
        Args:
            d_model: 输入/输出维度
            dim_feedforward: 中间层维度（通常为d_model的4倍）
            dropout: 随机失活比例
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        """前向传播"""
        x = F.relu(self.linear1(x))  # ReLU激活
        x = self.dropout(x)
        return self.linear2(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层（包含自注意力+前馈网络）"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """编码器层前向传播"""
        # 自注意力子层
        src2 = self.self_attn(src, src, src, src_mask)  # Q=K=V=src
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 层归一化

        # 前馈网络子层
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)  # 残差连接
        return self.norm2(src)  # 层归一化


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层（包含自注意力+交叉注意力+前馈网络）"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)  # 自注意力
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)  # 编码器-解码器注意力
        self.ffn = FeedForward(d_model, dim_feedforward, dropout)
        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """解码器层前向传播
        Args:
            tgt: 解码器输入 [batch_size, tgt_len, d_model]
            memory: 编码器输出 [batch_size, src_len, d_model]
        """
        # 自注意力子层
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)  # Q=K=V=tgt
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 编码器-解码器注意力子层
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)  # Q=tgt, K=V=memory
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 前馈网络子层
        tgt2 = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        return self.norm3(tgt)


class Transformer(nn.Module):
    """完整的Transformer模型（编码器-解码器架构）"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # 词嵌入层
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # 编码器堆叠
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 解码器堆叠
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        Returns:
            预测结果 [batch_size, tgt_len, tgt_vocab_size]
        """
        # 嵌入层 + 位置编码
        src = self.dropout(self.pos_encoder(self.src_embed(src)))
        tgt = self.dropout(self.pos_encoder(self.tgt_embed(tgt)))

        # 编码器前向传播
        for layer in self.encoder:
            src = layer(src, src_mask)

        # 解码器前向传播
        for layer in self.decoder:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        # 输出层
        return self.fc_out(tgt)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    SRC_VOCAB_SIZE = 5000
    TGT_VOCAB_SIZE = 5000
    d_model = 512
    nhead = 8
    num_layers = 6

    model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, d_model, nhead, num_layers, num_layers).to(device)

    # 虚拟输入
    src = torch.randint(0, SRC_VOCAB_SIZE, (32, 10)).to(device)  # batch_size=32, seq_len=10
    tgt = torch.randint(0, TGT_VOCAB_SIZE, (32, 20)).to(device)  # batch_size=32, seq_len=20

    output = model(src, tgt)
    print(output.shape)  # torch.Size([32, 20, 5000])
