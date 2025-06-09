#  -*- utf-8 -*-
import math
import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader
import torch.nn.functional as f


class simpletokenizer(object):
    def __init__(self):
        self.special_tokens = ['<pad>', '<bos>', '<eos>']
        self.vocab = {}
        self.inv_vocab = {}

    def build_vocab(self, texts, min_freq=1):
        # 统计所有文本中的词频
        freq = {}
        for text in texts:
            for tok in text.lower().split():
                freq[tok] = freq.get(tok, 0) + 1
        # 保留高频词与特殊符号
        tokens = self.special_tokens + [tok for tok, cnt in freq.items() if cnt >= min_freq]
        self.vocab = {tok: idx for idx, tok in enumerate(tokens)}
        self.inv_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def tokenize(self, text):
        return text.lower().split()

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(tok, self.vocab['<pad>']) for tok in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab.get(i, '<pad>') for i in ids]


class clmdataset(dataset):
    def __init__(self, texts, tokenizer, seq_len=16):
        examples = []
        pad_id = tokenizer.vocab['<pad>']
        bos_id = tokenizer.vocab['<bos>']
        eos_id = tokenizer.vocab['<eos>']

        for txt in texts:
            ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(txt))
            # 如果长度不足，pad对齐
            if len(ids) < seq_len + 1:
                ids += [pad_id] * (seq_len + 1 - len(ids))
            # 滑动窗口切片
            for i in range(len(ids) - seq_len):
                chunks = ids[i: i + seq_len + 1]  # 长度 seq_len + 1
                # 输入前加bos, 目标末尾加 eos
                inp = [bos_id] + chunks[:-1]
                tgt = chunks[1:] + [eos_id]
                examples.append((torch.tensor(inp), torch.tensor(tgt)))

        self.examples = examples
        self.seq_len = seq_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class decoderblock(nn.module):
    def __init__(self, embed_dim, num_heads, ff_hidden, dropout=0.1):
        super(decoderblock, self).__init__()
        self.self_attn = nn.multiheadattention(embed_dim, num_heads, dropout=dropout)
        self.lnl = nn.layernorm(embed_dim)
        self.ff = nn.sequential(
            nn.linear(embed_dim, ff_hidden),
            nn.gelu(),
            nn.linear(ff_hidden, embed_dim)
        )
        self.ln2 = nn.layernorm(embed_dim)
        self.drop = nn.dropout(dropout)

    def forward(self, x):
        # x: [t, b, d]
        t = x.size(0)
        # 因果编码：屏蔽未来位置
        mask = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        a, _ = self.self_attn(x, x, x, attn_mask=mask)  # 自注意力
        x = self.lnl(x + self.drop(a))  # 残差连接
        f = self.ff(x)  # 前向传播
        return self.ln2(x + self.drop(f))  # 残差连接 + 层归一化


class mingpt(nn.module):
    def __init__(self, vocab_size, seq_len=16, embed_dim=64, n_layers=2, num_heads=4, ff_hidden=256):
        super(mingpt, self).__init__()
        self.block_size = seq_len
        self.tok_emb = nn.embedding(vocab_size, embed_dim)
        self.pos_emb = nn.embedding(seq_len, embed_dim)
        self.layers = nn.modulelist([
            decoderblock(embed_dim, num_heads, ff_hidden) for _ in range(n_layers)
        ])
        self.ln_f = nn.layernorm(embed_dim)
        self.head = nn.linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [b, t]
        b, t = x.size()
        assert t <= self.block_size, "cannot forward, model block size is exhausted"
        tok = self.tok_emb(x)  # [b, t, d]
        pos = self.pos_emb(torch.arange(t, device=x.device))[none]  # [1, t, d]
        h = (tok + pos).transpose(0, 1)  # [t, b, d]
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h.transpose(0, 1))  # [b, t, d]
        return self.head(h)  # [b, t, v]


def train(model, dataloader, epochs=10, lr=1e-3, device='mps'):
    opt = torch.optim.adamw(model.parameters(), lr=lr)
    loss = nn.crossentropyloss()
    model.to(device).train()
    for epoch in range(epochs):
        total, acc = 0, 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            l = loss(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad()
            l.backward()
            opt.step()
            total += 1
            acc += l.item()
        print(f'epoch: {epoch} - loss: {acc/total}')


@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=50, strategy='greedy', temperature=1, **kwargs):
    """"""
    device = next(model.parameters()).device
    # 特殊tpkens的id列表
    special_ids = [tokenizer.vocab[tok] for tok in tokenizer.special_tokens]
    bos_id = tokenizer.vocab['<bos>']
    eos_id = tokenizer.vocab['<eos>']

    # 初始化输入：在prompt前加上<bos>
    prompt_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(prompt))
    x = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long, device=device)  # [1, l + 1]

    for _ in range(max_len):
        # 只保留最后的 block_szie 长度以匹配模型的输入限制
        x_cond = x if x.size(1) < model.block_size else x[:, -model.block_size:]
        # 前向，取最后一个时刻的logits
        logits = model(x_cond)[0, -1]  # [v] model(x_cond) ---> [b, t, v]
        # 屏蔽所有特殊tokens
        logits[special_ids] = -float('inf')
        # ==== 温度采样核心逻辑 ====
        if temperature != 1.0:
            logits = logits / temperature  # 缩放logits
        # 根据策略选择下一个token_id
        if strategy == 'greedy':
            idx = logits.argmax()
        elif strategy == 'top_k':
            probs, idxs = f.softmax(logits, dim=-1).topk(kwargs.get('top_k', 10))
            idx = idxs[probs.multinomial(num_samples=1)]
        elif strategy == 'top_p':
            probs, idxs = f.softmax(logits, dim=-1).sort(descending=True)
            cum = probs.cumsum(0)
            mask = cum < kwargs.get('top_p', 0.9)
            probs = probs * mask
            idx = idxs[probs.multinomial(num_samples=1)]
        else:
            raise ValueError(f'invalid strategy: {strategy}')
        # <eos>停止
        if idx.item() == eos_id:
            break
        # 拼接输入序列
        x = torch.cat([x, idx.unsqueeze(0)], dim=1)

    # 转回tokens 并进行过滤
    tokens = tokenizer.convert_ids_to_tokens(x[0].tolist())
    return [t for t in tokens if t not in tokenizer.special_tokens]


def test():
    texts = [
        "the quick brown fox jumps over the lazy dog"
        "hello how are you doing today",
        "this is a simple example sentence",
        "machine learning is changing the world",
        "what time is the meeting tomorrow",
        "please send me the report by friday",
        "the weather is really nice today",
        "i enjoy reading books on weekends",
        "let's grab coffee sometime next week",
        "python is my favorite programming language",
        "have you seen the latest movie",
        "my phone battery is almost dead",
        "the project deadline is approaching fast",
        "i need to buy groceries after work",
        "learning new skills is always beneficial",
        "can you help me with this problem",
        "the train was delayed by thirty minutes",
        "i love listening to music while working",
        "what are your plans for the weekend",
        "this restaurant has amazing food"
    ]
    tok = simpletokenizer()
    tok.build_vocab(texts)
    ds = clmdataset(texts, tok, seq_len=16)
    dl = dataloader(ds, batch_size=4, shuffle=true)

    device = 'mps'
    model = mingpt(len(tok.vocab), seq_len=18).to(device)
    train(model, dl, epochs=10, lr=1e-3, device=device)
    result = generate(model, tok, 'hello', max_len=20, strategy='top_p', top_k=5)
    print('生成结果： {}'.format(' '.join(result)))


if __name__ == '__main__':
    test()
    """
    epoch: 0 - loss: 4.2578620433807375
    epoch: 1 - loss: 2.726457357406616
    epoch: 2 - loss: 2.245575189590454
    epoch: 3 - loss: 2.0046085596084593
    epoch: 4 - loss: 1.7822702407836915
    epoch: 5 - loss: 1.6291230916976929
    epoch: 6 - loss: 1.4981262922286986
    epoch: 7 - loss: 1.378292942047119
    epoch: 8 - loss: 1.283841609954834
    epoch: 9 - loss: 1.206007719039917
    生成结果： reading doing with you seen world on python friday friday need reading approaching doing plans please you while the reading
    
    
    组合方式	适用场景	示例参数
    temperature=0.7, top_p=0.9	平衡质量与多样性（推荐）	GPT-3常用配置
    temperature=1.2, sample	创意生成	诗歌/故事生成
    temperature=0.3, top_k=50	事实性回答	问答系统
    """