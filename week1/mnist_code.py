from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def prepare_data():
    # 定义数据转换（归一化）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为 Tensor，并归一化到 [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # 标准化到 [-1, 1]
    ])

    # 加载训练集和测试集
    train_data = datasets.MNIST(
        root='./../data',  # 数据存储路径
        train=True,     # 加载训练集
        download=True,  # 如果本地没有，则下载
        transform=transform
    )

    test_data = datasets.MNIST(
        root='./../data',
        train=False,    # 加载测试集
        download=True,
        transform=transform
    )

    print("训练集样本数:", len(train_data))
    print("测试集样本数:", len(test_data))
    print("图片形状:", train_data[0][0].shape)  # torch.Size([1, 28, 28])
    # print("类别标签:", train_data[0][1])  # 数字 0~9

    return train_data, test_data


def plt_data(data, ix):
    image, label = data[ix]  # 第 1 个样本
    image = image.squeeze()  # 去掉 batch 维度 (1,28,28) -> (28,28)

    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label}")
    plt.show()


if __name__ == '__main__':
    train_data, test_data = prepare_data()
    ix = 19213
    plt_data(train_data, ix)
    print("查看 train_data: {}".format(ix))
    print("类别标签:", train_data[ix][1])  # 数字 0~9