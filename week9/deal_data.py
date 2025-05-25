#  -*- utf-8 -*-

import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


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


def encode_sentence(sentence, tokenize_fn, word2idx, max_len):
    # 获取句子的词编码
    tokens = tokenize_fn(sentence)
    # print(word2idx)
    token_ids = [word2idx['<BOS>']] + [word2idx.get(token, '<UNK>') for token in tokens] + [word2idx['<EOS>']]
    if len(token_ids) < max_len:
        token_ids.extend([word2idx['<PAD>']] * (max_len - len(token_ids)))  # 填充
    else:
        token_ids = token_ids[:max_len]
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

def test():
    data_obj = MyData(PARALLEL_DATA)


if __name__ == '__main__':
    test()
