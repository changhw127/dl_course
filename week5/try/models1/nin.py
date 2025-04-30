import torch
import torch.nn as nn
import torch.optim as optim


class NinBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.0):
        super(NinBlock, self).__init__()
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
    def __init__(self):
        super(MyNIN, self).__init__()
        self.m_name = 'NIN_Optimized'
        self.features = nn.Sequential(
            NinBlock(3, 32, kernel_size=5, stride=1, padding=2, dropout=0.1),   # 100x100
            nn.MaxPool2d(2, 2),                                                 # 50x50

            NinBlock(32, 64, kernel_size=5, stride=1, padding=2, dropout=0.2),  # 50x50
            nn.MaxPool2d(2, 2),                                                 # 25x25

            NinBlock(64, 128, kernel_size=3, stride=1, padding=1, dropout=0.3), # 25x25
            nn.MaxPool2d(2, 2),                                                 # 12x12

            NinBlock(128, 128, kernel_size=3, stride=1, padding=1, dropout=0.4),# 12x12
            nn.MaxPool2d(2, 2),                                                 # 6x6
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        self.lr = 0.001
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        输入3x100x100
        NIN结构优化版，适合小尺寸输入
        通道数减少，层数适中
        每层后加BN和Dropout
        多次MaxPool
        使用AdaptiveAvgPool2d和Dropout
        适合2分类
        
        通道数为32-64-128，参数量大幅减少。
        每个NIN block后有BN和Dropout，提升泛化。
        池化步长为2，特征图不会过早变小。
        最后全局平均池化+Dropout+全连接，防止过拟合。
        结构和你原有风格一致，方便直接替换。
        '''.replace('        ', '')
        return desc

    def init_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
        return True

    def init_optimizer(self):
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


class NINBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_channels):
        super(NINBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, mlp_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mlp_channels, mlp_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MyNIN0(nn.Module):
    def __init__(self):
        super(MyNIN0, self).__init__()
        self.m_name = 'NIN'
        self.features = nn.Sequential(
            NINBlock(3, 96, 96),
            nn.MaxPool2d(3, stride=2, padding=1),   # 100->50
            nn.Dropout(0.5),

            NINBlock(96, 256, 256),
            nn.MaxPool2d(3, stride=2, padding=1),   # 50->25
            nn.Dropout(0.5),

            NINBlock(256, 384, 384),
            nn.MaxPool2d(3, stride=2, padding=1),   # 25->13
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            # NIN最后一层通常用1x1卷积输出类别数
            nn.Conv2d(384, 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
        )
        self.lr = 0.001
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        输入3x100x100
        采用NIN结构
        每个Block由NINBlock类实现
        Block结构: Conv3x3+ReLU+Conv1x1+ReLU+Conv1x1+ReLU
        多次MaxPool
        使用Dropout
        最后1x1卷积输出类别数
        使用全局平均池化
        适合2分类
        '''.replace('        ', '')
        return desc

    def init_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
        return True

    def init_optimizer(self):
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