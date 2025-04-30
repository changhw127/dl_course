# -*- utf-8 -*-

import os
import pandas as pd

folder_path = 'results'  # 替换为你的文件夹路径
file_list = os.listdir(folder_path)
print(file_list)

columns = [
    'Adam_32_0.01',
    'Adam_32_0.001',
    'Adam_64_0.01',
    'Adam_64_0.001',
    'SGD_32_0.01',
    'SGD_32_0.001',
    'SGD_64_0.01',
    'SGD_64_0.001',
]
index = ['lenet', 'alexnet', 'nin', 'vgg', 'googlenet', 'resnet']
df1 = pd.DataFrame(columns=columns, index=index)
df1 = df1.fillna('0')
df2 = pd.DataFrame(columns=columns, index=index)
df2 = df2.fillna('0')

for name in file_list:
    if '.log' in name:
        m, opt, lr, b = name.replace('.log', '').split('_')
        ls = open(f'{folder_path}/{name}').readlines()
        acc = ls[-2].strip().split(':')[-1].strip()
        cost = ls[-3].strip().split(' ')[1]
        df1.loc[m, f'{opt}_{b}_{lr}'] = acc
        df2.loc[m, f'{opt}_{b}_{lr}'] = cost

df1.to_csv('results/result_acc.csv')
df2.to_csv('results/result_cost.csv')