import torch
import torch.nn as nn
import torch.optim as optim


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, c1x1, c3x3_reduce, c3x3, c5x5_reduce, c5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, c1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, c3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3x3_reduce, c3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # 1x1 conv -> 5x5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5x5_reduce, c5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], 1)


class MyGoogLeNet(nn.Module):
    def __init__(self):
        super(MyGoogLeNet, self).__init__()
        self.m_name = 'GoogLeNet_O'
        # 更小的初始卷积核和通道数
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   # 100x100
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 100x100
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                              # 50x50

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 50x50
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),                              # 25x25

            # Inception blocks (参数减少)
            InceptionBlock(64, 32, 32, 48, 8, 16, 16),              # out: 112
            InceptionBlock(112, 48, 32, 64, 8, 16, 16),             # out: 144
            nn.MaxPool2d(2, stride=2),                              # 12x12

            InceptionBlock(144, 64, 48, 96, 12, 24, 24),            # out: 208
            InceptionBlock(208, 64, 48, 96, 12, 24, 24),            # out: 208
            nn.MaxPool2d(2, stride=2),                              # 6x6

            InceptionBlock(208, 96, 64, 128, 16, 32, 32),           # out: 288
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(288, 2)
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
        GoogLeNet结构优化版，适合小尺寸输入
        初始卷积核和通道数减少
        Inception Block参数减少
        堆叠层数减少
        多次MaxPool
        使用AdaptiveAvgPool2d和Dropout
        
        初始卷积核：由 7x7/stride=2 改为 3x3/stride=1，通道数减少，适合小图像。
        Inception Block：每个分支的通道数减少，参数量更小。
        Inception Block 堆叠数：减少为 5 个，防止网络过深。
        池化：每次池化后尺寸减半，最终特征图为 6x6，最后全局平均池化。
        Dropout：增加到 0.5，防止过拟合。
        全连接层：输入通道数为最后一个 Inception Block 的输出通道数。
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


class MyGoogLeNet0(nn.Module):
    def __init__(self):
        super(MyGoogLeNet0, self).__init__()
        self.m_name = 'GoogLeNet'
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 100->50
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),                  # 50->25
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),                  # 25->13

            # Inception blocks
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),          # out: 256
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),        # out: 480
            nn.MaxPool2d(3, stride=2, padding=1),                  # 13->7

            InceptionBlock(480, 192, 96, 208, 16, 48, 64),         # out: 512
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),        # out: 512
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),        # out: 512
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),        # out: 528
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),      # out: 832
            nn.MaxPool2d(3, stride=2, padding=1),                  # 7->4

            InceptionBlock(832, 256, 160, 320, 32, 128, 128),      # out: 832
            InceptionBlock(832, 384, 192, 384, 48, 128, 128),      # out: 1024
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, 2)
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
        采用GoogLeNet(Inception v1)结构
        每个Inception Block由InceptionBlock类实现
        多次MaxPool
        使用AdaptiveAvgPool2d
        使用Dropout
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