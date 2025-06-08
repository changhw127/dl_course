#  -*- utf-8 -*-

import re
import random
import torch
from torch.utils.data import Dataset


vocab =  ['<PAD>', '<CLS>', '<SEP>', '<MASK>', '<UNK>',
          'i', 'like', 'cats', 'you', 'do', 'not', 'know', 'love', 'playing', 'football', '.', '!', '?', ':', ','
          ]

token2id = {token: idx for idx, token in enumerate(vocab)}
id2token =  {idx: token for idx, token in enumerate(vocab)}


def basic_tokenizer(text):
    # 小写 + 保留符合
    text = text.lower()
    tokens = re.findall(r"[a-z]+|[.,!?]:", text)
    return tokens

def pad_or_truncate(seq, pad_value, max_len):
     return seq[:max_len] + [pad_value] * max(max_len - len(seq), 0)

def encode(tokens, max_len=12):
    token_ids = [token2id.get(token, token2id['<UNK>']) for token in tokens]
    token_ids = [token2id['<CLS>']] + token_ids + [token2id['<SEP>']]
    return pad_or_truncate(token_ids, token2id['<PAD>'], max_len)


class BertPretrainDataset(Dataset):
    def __init__(self, sentence_pairs, tokenizer, mask_prob=0.15, max_len=12):
        self.pairs = sentence_pairs
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def random_mask(self, token_ids):
        output = token_ids.copy()
        labels = [-100] * len(token_ids)
        for i in range(1, len(token_ids) - 1):
            if random.random() < self.mask_prob:
                labels[i] = token_ids[i]
                prob = random.random()
                if prob < 0.8:
                    output[i] = token2id['<MASK>']
                elif prob < 0.9:
                    output[i] = random.randint(5, len(token2id) - 1)
                else:
                    output[i] = token_ids[i]

    def __getitem__(self, index):
        A, B, is_next = self.pairs[index]
        all_tokens = self.tokenizer(A) + ['<SEP>'] + self.tokenizer(B)







def test():
    a = 'i like cats.'
    b = 'do you know?'
    a = basic_tokenizer(a)
    b = basic_tokenizer(b)
    all_tokens = a + ['<SEP>'] + b
    print(encode(all_tokens))
    # [1, 5, 6, 7, 2, 9, 8, 11, 2, 0, 0, 0]



if __name__ == '__main__':
    test()














