# -*- utf-8 -*-

"""
1 把训练数据拼成一个csv
2 测试数据做成csv，用于生产出预测结果
"""

import os
import pandas as pd
from collections import Counter

def list_files(directory):
    # 获取目录下的所有文件和文件夹名称
    items = os.listdir(directory)
    # 过滤出文件
    files = [f for f in items if os.path.isfile(os.path.join(directory, f))]
    return files


def deal_data():
    records = []
    path = '../data/a-i-2025-02/train/train/neg/'
    files = list_files(path)
    for file in files:
        c = open(path + file, 'r', encoding='gbk', errors='ignore').readlines()
        s = ' '.join([i.strip() for i in c])
        s = s.replace('content>', '')
        s = s.strip()
        idx = file.replace('(', '').replace(')', '').replace(' ', '_').replace('.txt', '')
        records.append(
            {
                'ID': idx,
                'Label': 0,
                'Content': s,
            }
        )
    path = '../data/a-i-2025-02/train/train/pos/'
    files = list_files(path)
    for file in files:
        c = open(path + file, 'r', encoding='gbk', errors='ignore').readlines()
        s = ' '.join([i.strip() for i in c])
        s = s.replace('content>', '')
        s = s.strip()
        idx = file.replace('(', '').replace(')', '').replace(' ', '_').replace('.txt', '')
        records.append(
            {
                'ID': idx,
                'Label': 1,
                'Content': s,
            }
        )
    df = pd.DataFrame.from_records(records)
    df = df.fillna('')
    df.to_csv('result/data.csv', index=None)
    return


def deal_test_data():
    records = []
    path = '../data/a-i-2025-02/test/test/'
    files = list_files(path)
    for file in files:
        c = open(path + file, 'r', encoding='gbk', errors='ignore').readlines()
        s = ' '.join([i.strip() for i in c])
        s = s.replace('content>', '')
        s = s.strip()
        idx = file.replace('(', '').replace(')', '').replace(' ', '_').replace('.txt', '')
        records.append(
            {
                'ID': idx,
                'IntID': int(idx),
                'Label': 0,
                'Content': s,
            }
        )
    df = pd.DataFrame.from_records(records)
    df = df.fillna('')
    df.to_csv('result/pred_data.csv', index=None)
    return


def analysis_data():
    df = pd.read_csv('result/data.csv')
    df = df.fillna('')
    df2 = pd.read_csv('result/pred_data.csv')
    df2 = df2.fillna('')
    ls1 = list(df['Content'])
    ls2 = list(df2['Content'])
    x = []
    for ix in range(len(ls1)):
        x.append(len(ls1[ix]))
    print(max(x))
    print(Counter(x).most_common(20))
    a = [x for x in Counter(x).most_common() if x[0] > 512]

    x = []
    for i in ls2:
        x.append(len(i))
    print(max(x))
    print(Counter(x).most_common(20))
    # 3008
    # 3008
    # 3008
    # [(38, 96), (39, 90), (37, 85), (42, 84), (43, 83), (41, 81), (50, 78), (46, 75), (44, 74), (40, 72), (47, 71), (52, 69), (49, 67), (45, 67), (36, 67), (34, 66), (54, 66), (33, 65), (51, 64), (53, 63)]
    # 3008
    # [(38, 98), (46, 86), (42, 84), (39, 79), (50, 76), (43, 74), (40, 73), (51, 70), (37, 70), (49, 69), (41, 69), (48, 68), (29, 68), (33, 68), (54, 67), (32, 67), (45, 67), (53, 66), (36, 64), (31, 64)]
    return