# -*- utf-8 -*-
import torch.nn as nn
import torch.optim as optim


class MyLeNet(nn.Module):
    def __init__(self, lr=0.01, opt='SGD'):
        super(MyLeNet, self).__init__()
        self.m_name = 'LeNet'
        self.raw_size = 32
        self.lr = lr
        self.opt = opt
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 2),
        )

        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        使用原生架构,仅修改输入size，和输出类别为2类
        '''.replace('        ', '').strip()
        return desc

    def init_criterion(self):
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        return True

    def init_optimizer(self):
        # 定义优化器
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

