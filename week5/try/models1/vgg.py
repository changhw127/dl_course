import torch
import torch.nn as nn
import torch.optim as optim

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, dropout=0.0):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)


class MyVGG(nn.Module):
    def __init__(self):
        super(MyVGG, self).__init__()
        self.m_name = 'VGG_Optimized'
        self.features = nn.Sequential(
            VGGBlock(3, 32, 2, dropout=0.1),   # 100x100
            nn.MaxPool2d(2, 2),                # 50x50

            VGGBlock(32, 64, 2, dropout=0.2),  # 50x50
            nn.MaxPool2d(2, 2),                # 25x25

            VGGBlock(64, 128, 2, dropout=0.3), # 25x25
            nn.MaxPool2d(2, 2),                # 12x12

            VGGBlock(128, 128, 2, dropout=0.4),# 12x12
            nn.MaxPool2d(2, 2),                # 6x6
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
        VGG结构优化版，适合小尺寸输入
        通道数减少，层数适中
        每层后加BN和Dropout
        多次MaxPool
        使用AdaptiveAvgPool2d和Dropout
        适合2分类
        
        通道数为32-64-128，参数量大幅减少。
        每个VGG block后有BN和Dropout，提升泛化。
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


class MyVGG0(nn.Module):
    def __init__(self):
        super(MyVGG0, self).__init__()
        self.m_name = 'VGG11'
        self.features = nn.Sequential(
            VGGBlock(3, 64, 1),      # Block 1: 1 conv
            VGGBlock(64, 128, 1),    # Block 2: 1 conv
            VGGBlock(128, 256, 2),   # Block 3: 2 convs
            VGGBlock(256, 512, 2),   # Block 4: 2 convs
            VGGBlock(512, 512, 2),   # Block 5: 2 convs
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2),
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
        采用VGG11结构
        每个Block由VGGBlock类实现
        每个卷积后有BN和ReLU
        5个MaxPool
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