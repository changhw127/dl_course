import torch
import torch.nn as nn
import torch.optim as optim


class NINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.0):
        super(NINBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.block(x)


class MyNIN(nn.Module):
    def __init__(self, lr=0.01, opt='SGD'):
        super(MyNIN, self).__init__()
        self.m_name = 'NIN'
        self.raw_size = 32  # NIN原始输入尺寸
        self.lr = lr
        self.opt = opt
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.features = nn.Sequential(
            NINBlock(3, 192, kernel_size=5, stride=1, padding=2),  # 32x32
            nn.MaxPool2d(3, stride=2, padding=1),                  # 16x16
            nn.Dropout(0.5),
            NINBlock(192, 160, kernel_size=5, stride=1, padding=2),# 16x16
            nn.MaxPool2d(3, stride=2, padding=1),                  # 8x8
            nn.Dropout(0.5),
            NINBlock(160, 96, kernel_size=3, stride=1, padding=1), # 8x8
            nn.MaxPool2d(3, stride=2, padding=1),                  # 4x4
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            # NIN最后一层用1x1卷积代替全连接
            nn.Conv2d(96, 2, kernel_size=1),  # 输出2类
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),     # 全局平均池化
            nn.Flatten()
        )
        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        NIN结构，输入32x32，输出2类
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
        x = self.features(x)
        x = self.classifier(x)
        return x

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