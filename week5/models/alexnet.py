# -*- utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MyAlexNet(nn.Module):
    def __init__(self, lr=0.01, opt='SGD'):
        super(MyAlexNet, self).__init__()
        self.m_name = 'AlexNet'
        self.raw_size = 224  # AlexNet原始输入尺寸
        self.lr = lr
        self.opt = opt
        self.criterion = None
        self.optimizer = None
        self.loss = None
        # AlexNet原始结构，输出2类
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 224 -> 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 55 -> 27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),           # 27 -> 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 27 -> 13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),          # 13 -> 13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),          # 13 -> 13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),          # 13 -> 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 13 -> 6
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2),  # 输出2类
        )

        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        AlexNet原始结构，输入224x224，输出2类
        '''.replace('        ', '').strip()
        return desc

    def init_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
        return True

    def init_optimizer(self):
        if self.opt == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        if self.opt == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.optimizer.zero_grad()
        return True

    def forward(self, x):
        return self.model(x)

    def calc_loss(self, y, Y, save_loss=True):
        loss = self.criterion(y, Y)
        if save_loss:
            self.loss = loss
        return loss

    def backward(self):
        self.loss.backward()
        return True

    def update_params(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        return True