import math
import torch
import torch.nn as nn
from deal_data import MyData, PARALLEL_DATA, encode_sentence, tokenize_en
from utlis.globals import Logger, save_model, load_model, get_device, plt_loss_result


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_item = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_item)  # 偶数sin
        pe[:, 1::2] = torch.cos(position * div_item)  # 奇数cos
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0  # 模型维度必须整除头数

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        # 线性层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, d_model)

        # 用于可视化权重
        self.attention_weights = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # 线性映射
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        # 分多头 reshape:  (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 计算缩放点注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: (batch_size, num_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # 保存权重
        self.attention_weights = attn.detach().cpu().numpy()

        out = torch.matmul(attn, V)
        # 合并多个头
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.linear_out(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(ff_out + self.dropout2(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        # mask: (batch_size, seq_len, seq_len)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        cross_attn_out = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 num_encoder_layers, num_decoder_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout=dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout=dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)  # 输出层

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src tensor([[ 1, 11, 12, 13,  6,  2,  0]], device='mps:0')
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        # torch.Size([1, 7, 64]) 将输入的src长度为7个单词，64表示词向量维度，编码后，进行缩放
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb, mask=src_mask)  # torch.Size([1, 7, 64])

        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)  # torch.Size([1, 6]) -> torch.Size([1, 6, 64])
        tgt_emb = self.pos_encoder(tgt_emb)
        output = self.decoder(tgt_emb, memory, tgt_mask)  # torch.Size([1, 6, 64])
        logits = self.fc_out(output)  # torch.Size([1, 6, 25])
        return logits

    @classmethod
    def generate_square_subsequent_mask(cls, sz):
        mask = torch.tril(torch.ones(sz, sz)).bool()
        return mask


class SimpleLRScheduler(object):
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num > self.warmup_steps:
            lr = self.base_lr * self.step_num / float(self.warmup_steps)
        else:
            lr = self.base_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def train_model():
    # 参数设置
    data_obj = MyData(PARALLEL_DATA)
    m_name = 'my_transformer_details'
    logger = Logger('results', m_name)
    logger.log(f'{m_name} train --------------', clear=True)

    d_model = 64
    num_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_ff = 128
    dropout = 0.1
    src_vocab_size = len(data_obj.vocab_en)
    tgt_vocab_size = len(data_obj.vocab_zh)
    model_args = {
        'd_model': d_model,
        'num_heads': num_heads,
        'num_encoder_layers': num_encoder_layers,
        'num_decoder_layers': num_decoder_layers,
        'd_ff': d_ff,
        'dropout': dropout,
        'src_vocab_size': src_vocab_size,
        'tgt_vocab_size': tgt_vocab_size,
    }

    device = get_device(True, logger)
    logger.log('Parameters: {}'.format(locals()))
    logger.log('Data: {}'.format(data_obj.__dict__))
    logger.log('model_args: {}'.format(str(model_args)))

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads,
                        num_encoder_layers, num_decoder_layers, d_ff, dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=data_obj.word2idx_zh['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = SimpleLRScheduler(optimizer, warmup_steps=4000, base_lr=1e-3)

    epochs = 50
    loss_ls = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for src_sentence, tgt_sentence in zip(data_obj.encoded_en, data_obj.encoded_zh):
            # src_sentence tensor([ 1, 11, 12, 13,  6,  2,  0])
            # tgt_sentence tensor([1, 7, 8, 2, 0, 0, 0])
            src_input = src_sentence.unsqueeze(0).to(device)  # tensor([[ 1, 11, 12, 13,  6,  2,  0]], device='mps:0')
            tgt_input = tgt_sentence[:-1].unsqueeze(0).to(device)  # tensor([[1, 7, 8, 2, 0, 0]], device='mps:0')
            tgt_output = tgt_sentence[1:].unsqueeze(0).to(device)  # tensor([[7, 8, 2, 0, 0, 0]], device='mps:0')

            tgt_seq_len = tgt_input.size(1)  # 6
            tgt_mask = Transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt_input.device).to(dtype=torch.float32)
            # tensor([[1., 0., 0., 0., 0., 0.],
            #         [1., 1., 0., 0., 0., 0.],
            #         [1., 1., 1., 0., 0., 0.],
            #         [1., 1., 1., 1., 0., 0.],
            #         [1., 1., 1., 1., 1., 0.],
            #         [1., 1., 1., 1., 1., 1.]], device='mps:0')

            logits = model(src_input, tgt_input, src_mask=None, tgt_mask=tgt_mask)  # torch.Size([1, 6, 25])
            logits = logits.reshape(-1, tgt_vocab_size)  # torch.Size([6, 25])
            tgt_output = tgt_output.reshape(-1)  # torch.Size([6])

            loss = criterion(logits, tgt_output)
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            total_loss += loss.item()

        logger.log(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_obj.encoded_en):.4f}')
        loss_ls.append(total_loss / len(data_obj.encoded_en))
    logger.log("Training finished!")
    save_model(model, m_name)
    plt_loss_result(loss_ls, path='results', name=m_name, logger=logger)
    return True


def test_model():
    src_sentence = PARALLEL_DATA[2][0]
    device = get_device()
    data_obj = MyData(PARALLEL_DATA)
    max_len = 20

    model_args = {'d_model': 64, 'num_heads': 4, 'num_encoder_layers': 2, 'num_decoder_layers': 2, 'd_ff': 128, 'dropout': 0.1, 'src_vocab_size': 27, 'tgt_vocab_size': 29}
    model = load_model(Transformer, 'results/my_transformer_details.pth', 'mps', **model_args)
    model.eval()

    print('输入:', src_sentence)
    src_ids = encode_sentence(src_sentence, tokenize_en, data_obj.word2idx_en, data_obj.max_len_en)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    tgt_ids = [data_obj.word2idx_zh['<BOS>']]
    for i in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(device)
        tgt_mask = Transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(src_tensor.device).to(dtype=torch.float32)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)
        next_token = output[0, -1, :].argmax().item()
        tgt_ids.append(next_token)
        if next_token == data_obj.word2idx_zh['<EOS>']:
            break
    translated_sentence = [data_obj.idx2word_zh[idx] for idx in tgt_ids]
    print('预测:' + ' '.join(translated_sentence))
    """
    模型参数已从 results/my_transformer.pth 加载
    输入: I am a student.
    预测:<BOS> 我 是 一名 学生。 <EOS>
    """
    return


if __name__ == '__main__':
    train_model()
    test_model()
