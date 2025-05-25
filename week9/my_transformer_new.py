import math
import torch
import torch.nn as nn
from deal_data import MyData, PARALLEL_DATA, tokenize_en, encode_sentence
from utlis.globals import Logger, save_model, plt_loss_result, get_device, load_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, is_decoder=False):
        super(TransformerLayer, self).__init__()
        self.is_decoder = is_decoder

        # 使用 PyTorch 内置的 MultiheadAttention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        if is_decoder:
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if is_decoder:
            self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if is_decoder:
            self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory=None, src_mask=None, tgt_mask=None):
        # 调整输入形状为 (seq_len, batch_size, d_model)
        x_ = x.transpose(0, 1)

        # 自注意力
        attn_output, _ = self.self_attn(x_, x_, x_, attn_mask=tgt_mask)
        attn_output = self.dropout1(attn_output)
        x = x + attn_output.transpose(0, 1)
        x = self.norm1(x)

        # 如果是解码器，还需要交叉注意力
        if self.is_decoder and memory is not None:
            memory_ = memory.transpose(0, 1)
            cross_output, _ = self.cross_attn(x_, memory_, memory_, attn_mask=src_mask)
            cross_output = self.dropout2(cross_output)
            x = x + cross_output.transpose(0, 1)
            x = self.norm2(x)

        # 前馈网络
        ffn_output = self.ffn(x)
        ffn_output = self.dropout3(ffn_output) if self.is_decoder else self.dropout2(ffn_output)
        x = x + ffn_output
        x = self.norm3(x) if self.is_decoder else self.norm2(x)

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads,
                 num_encoder_layers, num_decoder_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model

        # 嵌入层
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # 编码器和解码器
        self.encoder = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout, is_decoder=False)
            for _ in range(num_encoder_layers)
        ])

        self.decoder = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout, is_decoder=True)
            for _ in range(num_decoder_layers)
        ])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器部分
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        memory = src_emb
        for layer in self.encoder:
            memory = layer(memory, src_mask=src_mask)

        # 解码器部分
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        output = tgt_emb
        for layer in self.decoder:
            output = layer(output, memory=memory, tgt_mask=tgt_mask)

        logits = self.fc_out(output)
        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def train_model():
    # 参数设置
    data_obj = MyData(PARALLEL_DATA)
    m_name = 'pytorch_transformer'
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

    device = get_device(True, logger)

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads,
                        num_encoder_layers, num_decoder_layers, d_ff, dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=data_obj.word2idx_zh['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    epochs = 50
    loss_ls = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for src_sentence, tgt_sentence in zip(data_obj.encoded_en, data_obj.encoded_zh):
            src_input = src_sentence.unsqueeze(0).to(device)
            tgt_input = tgt_sentence[:-1].unsqueeze(0).to(device)
            tgt_output = tgt_sentence[1:].unsqueeze(0).to(device)

            tgt_seq_len = tgt_input.size(1)
            tgt_mask = Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)

            optimizer.zero_grad()

            logits = model(src_input, tgt_input, tgt_mask=tgt_mask)
            logits = logits.view(-1, tgt_vocab_size)
            tgt_output = tgt_output.view(-1)

            loss = criterion(logits, tgt_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_obj.encoded_en)
        loss_ls.append(avg_loss)
        logger.log(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

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
    model = load_model(Transformer, 'results/pytorch_transformer.pth', 'mps', **model_args)

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

    return


if __name__ == '__main__':
    train_model()
    test_model()
    """"
    改进点
    1 使用 nn.MultiheadAttention：替换了自定义的注意力实现，使用 PyTorch 内置的高效实现。
    2 简化 Transformer 层结构：将编码器和解码器层合并为一个可配置的 TransformerLayer 类，通过 is_decoder 参数区分。
    3 优化训练过程：
        移除了自定义的学习率调度器，使用固定学习率
        简化了损失计算和反向传播代码
        保持了相同的输入输出接口，确保与原有代码兼容
    4 注意力掩码生成：使用 PyTorch 推荐的掩码生成方式，确保正确性。
    """