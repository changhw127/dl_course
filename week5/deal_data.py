# -*- utf-8 -*-
import torch
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit


class MyData(object):
    def __init__(self, path, batch_size, resize=100, test_ratio=0.2, random_state=2,
                 is_subset=False, subset_percent=0.2, logger=None):
        self.path = path
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.full_dataset = None
        self.random_state = random_state
        self.is_subset = is_subset
        self.subset_percent = subset_percent
        self.logger = logger
        self.resize = resize
        self._load_data()

    def img_show(self, dataset, index):
        # 查看数据
        # dataset[0]
        data = dataset[index]
        image, label = data
        plt.imshow(image.permute(1, 2, 0))  # 从(C,H,W)转为(H,W,C)
        plt.axis('off')
        plt.show()
        print(f"标签: {self.full_dataset.classes[label]}")
        return True

    def _transform(self):
        # 定义图像预处理流程
        transform = transforms.Compose([
            # 1. 先调整大小：将图像的短边缩放，长边按比例缩放
            transforms.Resize(self.resize + 2),
            # 2. 然后中心裁剪：确保所有图像都是指定size
            transforms.CenterCrop(self.resize),
            # 3. 转换为Tensor并自动归一化到[0,1]
            transforms.ToTensor(),
            # 可选：标准化（使用ImageNet的均值和标准差）
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def _load_data(self):
        self.full_dataset = datasets.ImageFolder(
            root=self.path,
            transform=self._transform()
        )
        labels = [label for _, label in self.full_dataset]

        # 使用分层抽样划分
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.random_state)
        train_indices, test_indices = next(sss.split(np.zeros(len(labels)), labels))
        self.train_dataset = torch.utils.data.Subset(self.full_dataset, train_indices)
        self.test_dataset = torch.utils.data.Subset(self.full_dataset, test_indices)

        if self.is_subset:
            # 默认获取 1% 的数据用于拉通
            indices = torch.randperm(len(self.train_dataset))[:int(len(self.train_dataset) * self.subset_percent)]
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)
            indices = torch.randperm(len(self.test_dataset))[:int(len(self.test_dataset) * self.subset_percent)]
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, indices)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 训练集需要打乱
            pin_memory=True  # 加速GPU传输
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1000,
            shuffle=False,  # 测试集不需要打乱
        )
        msg = f"train_data: {len(self.train_loader.dataset)}, test_data: {len(self.test_loader.dataset)}, batch_size: {self.batch_size}"
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg)
        return True


def test():
    path = '../data/course/faces'
    my_data = MyData(path, batch_size=64, test_ratio=0.2, random_state=6, is_subset=True, subset_percent=0.2)
    train_data, test_data, train_loader, test_loader = my_data.train_dataset, my_data.test_dataset, my_data.train_loader, my_data.test_loader
    print(f"train_data: {len(train_loader.dataset)}, test_data: {len(test_loader.dataset)}, batch_size: {my_data.batch_size}")
    idx = 111
    my_data.img_show(train_data, idx)
    print(train_data[idx][0].shap)
    # 展示图像，展示标签
