#  -*- utf-8 -*-

import math
import re
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader

SPECIAL_TOKENS = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
# PARALLEL_DATA = [
#     ('Hello, how ard you?', '你好，你 怎么样？'),
#     ('I am fine, thank you.', '我 很好，谢谢。'),
#     ('I am a student.', '我 是 一名 学生。'),
#     ('I love learning new things', '我 爱 学习 新 事物。'),
#     ('This is a book.', '这 是 一 本 书'),
#     ('Thank you very much.', '非常 感谢 你。'),
#     ('I am going to school.', '我 去 学校。'),
#     ('I am going to work.', '我 去 工作。'),
# ]
PARALLEL_DATA = [
    ('Hello, how ard you?', '你好 ，你 怎么样 ？'),
    ('I am fine, thank you.', '我 很好 ，谢谢 。'),
    ('I am a student.', '我 是 一名 学生 。'),
    ('I love learning new things', '我 爱 学习 新 事物 。'),
    ('This is a book.', '这 是 一 本 书'),
    ('Thank you very much.', '非常 感谢 你 。'),
    ('I am going to school.', '我 去 学校 。'),
    ('I am going to work.', '我 去 工作 。'),
]


def tokenize_en(sentence):
    sentence = sentence.lower()
    tokens = re.findall(r'[a-zA-Z]+|[,.!?]]', sentence)
    return tokens


def tokenize_zh(sentence):
    # 按照空格进行分词
    return sentence.split()


def build_vocab(sentences, tokenize_fn):
    """构建词汇表"""
    all_tokens = []
    max_len = 0
    for sentence in sentences:
        tokens = tokenize_fn(sentence)
        all_tokens.extend(tokens)
        max_len = max(max_len, len(tokens))

    counter = Counter(all_tokens)
    sorted_tokens = sorted(counter, key=counter.get, reverse=True)
    vocab = SPECIAL_TOKENS + sorted_tokens
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}

    return vocab, word2idx, idx2word, max_len


def encode_sentence(sentence, tokenize_fn, word2idx, max_len, to_tensor=False):
    # 获取句子的词编码
    tokens = tokenize_fn(sentence)
    # print(word2idx)
    token_ids = [word2idx['<BOS>']] + [word2idx.get(token, '<UNK>') for token in tokens] + [word2idx['<EOS>']]
    if len(token_ids) < max_len:
        token_ids.extend([word2idx['<PAD>']] * (max_len - len(token_ids)))  # 填充
    else:
        token_ids = token_ids[:max_len]
    if to_tensor:
        return torch.tensor(token_ids, dtype=torch.long)
    return token_ids


class MyData(object):
    def __init__(self, parallel_data):
        self.parallel_data = parallel_data
        self.vocab_en = None
        self.vocab_zh = None
        self.encoded_en = None
        self.encoded_zh = None
        self.encode_data = None
        self.word2idx_en = None
        self.word2idx_zh = None
        self.idx2word_en = None
        self.idx2word_zh = None
        self.max_len_en = None
        self.sentences_encoded()

    def sentences_encoded(self):
        # 获取所有英文句子的词
        parallel_data = self.parallel_data
        all_tokens_en = []
        for en, _ in parallel_data:
            all_tokens_en.extend(tokenize_en(en))
        counter_en = Counter(all_tokens_en)

        sorted_counter_en = sorted(counter_en, key=counter_en.get, reverse=True)  # 获取词频
        vocab_en = SPECIAL_TOKENS + sorted_counter_en  # 添加特殊字符
        word2idx_en = {word: idx for idx, word in enumerate(vocab_en)}
        idx2word_en = {idx: word for word, idx in word2idx_en.items()}

        # 中文单词处理
        all_tokens_zh = []
        for _, zh in parallel_data:
            all_tokens_zh.extend(tokenize_zh(zh))
        counter_zh = Counter(all_tokens_zh)
        sorted_counter_zh = sorted(counter_zh, key=counter_zh.get, reverse=True)
        vocab_zh = SPECIAL_TOKENS + sorted_counter_zh
        word2idx_zh = {word: idx for idx, word in enumerate(vocab_zh)}
        idx2word_zh = {idx: word for word, idx in word2idx_zh.items()}

        max_len_en = max(len(tokenize_en(en)) for en, _ in parallel_data) + 2
        max_len_zh = max(len(tokenize_zh(zh)) for _, zh in parallel_data) + 2

        encode_data = []
        for en, zh in parallel_data:
            en_ids = encode_sentence(en, tokenize_en, word2idx_en, max_len_en)
            zh_ids = encode_sentence(zh, tokenize_zh, word2idx_zh, max_len_zh)
            encode_data.append((en_ids, zh_ids))

        encoded_en = torch.tensor([pair[0] for pair in encode_data], dtype=torch.long)
        encoded_zh = torch.tensor([pair[1] for pair in encode_data], dtype=torch.long)

        # print('Encoded sentences: ', encoded_en)
        # print('Encoded sentences: ', encoded_zh)
        # 需要什么进行给值
        self.vocab_en = vocab_en
        self.vocab_zh = vocab_zh
        self.encode_data = encode_data
        self.encoded_en = encoded_en
        self.encoded_zh = encoded_zh
        self.word2idx_en = word2idx_en
        self.word2idx_zh = word2idx_zh
        self.idx2word_en = idx2word_en
        self.idx2word_zh = idx2word_zh
        self.max_len_en = max_len_en
        return True


class ParallelTextDataset(Dataset):
    def __init__(self, max_len=20):
        sentences_en = [pair[0] for pair in PARALLEL_DATA]
        sentences_zh = [pair[1] for pair in PARALLEL_DATA]
        vocab_en, word2idx_en, idx2word_en, max_len_en = build_vocab(sentences_en, tokenize_en)
        vocab_zh, word2idx_zh, idx2word_zh, max_len_zh = build_vocab(sentences_zh, tokenize_zh)

        self.vocab_en = vocab_en
        self.vocab_zh = vocab_zh
        self.word2idx_en = word2idx_en
        self.word2idx_zh = word2idx_zh
        self.idx2word_en = idx2word_en
        self.idx2word_zh = idx2word_zh
        self.max_len_en = max_len_en
        self.max_len_zh = max_len_zh
        self.sentences_en = sentences_en
        self.sentences_zh = sentences_zh
        self.tokenize_fn_en = tokenize_en
        self.tokenize_fn_zh = tokenize_zh
        self.max_len = max_len
        self.pad_idx_en = word2idx_en['<PAD>']
        self.pad_idx_zh = word2idx_zh['<PAD>']

    def __len__(self):
        return len(self.sentences_en)

    def __getitem__(self, idx):
        sentence_en = self.sentences_en[idx]
        sentence_zh = self.sentences_zh[idx]

        # 编码句子
        tensor_en = encode_sentence(sentence_en, self.tokenize_fn_en, self.word2idx_en, self.max_len, to_tensor=True)
        tensor_zh = encode_sentence(sentence_zh, self.tokenize_fn_zh, self.word2idx_zh, self.max_len, to_tensor=True)

        # 计算实际长度（不包括padding）
        # 确保比较结果是数值型张量
        en_len = torch.sum(tensor_en != self.pad_idx_en).item()
        zh_len = torch.sum(tensor_zh != self.pad_idx_zh).item()

        return {
            'src': tensor_en,
            'tgt': tensor_zh,
            'src_len': en_len,
            'tgt_len': zh_len
        }

    @staticmethod
    def collate_fn(batch, pad_idx):
        """自定义批处理函数，处理不同长度的句子"""
        src_batch = [item['src'] for item in batch]
        tgt_batch = [item['tgt'] for item in batch]
        src_len_batch = torch.tensor([item['src_len'] for item in batch])
        tgt_len_batch = torch.tensor([item['tgt_len'] for item in batch])

        # 自动padding
        src_batch = torch.nn.utils.rnn.pad_sequence(
            src_batch,
            batch_first=True,
            padding_value=pad_idx
        )
        tgt_batch = torch.nn.utils.rnn.pad_sequence(
            tgt_batch,
            batch_first=True,
            padding_value=pad_idx
        )

        return {
            'src': src_batch,
            'tgt': tgt_batch,
            'src_len': src_len_batch,
            'tgt_len': tgt_len_batch
        }


def test():
    data_obj = MyData(PARALLEL_DATA)


def test2():
    dataset = ParallelTextDataset(max_len=20)

    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=lambda batch: ParallelTextDataset.collate_fn(batch, pad_idx=dataset.pad_idx_en)
    )

    # 测试DataLoader
    for batch in dataloader:
        print("Source batch shape:", batch['src'].shape)
        print("Target batch shape:", batch['tgt'].shape)
        print("Source lengths:", batch['src_len'])
        print("Target lengths:", batch['tgt_len'])
        """
        Source batch shape: torch.Size([4, 20])
        Target batch shape: torch.Size([4, 20])
        Source lengths: tensor([7, 6, 6, 7])
        Target lengths: tensor([8, 6, 7, 6])
        """
        break


if __name__ == '__main__':
    # test()
    test2()
