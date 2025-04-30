import torch
import torch.nn as nn
import torch.optim as optim


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs, dropout=0.0):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class MyVGG(nn.Module):
    def __init__(self, lr=0.01, opt='SGD'):
        super(MyVGG, self).__init__()
        self.m_name = 'VGG_O'
        self.raw_size = 100  # 使用数据本身尺寸，优化了下，不然太慢了，原始为224
        self.lr = lr
        self.opt = opt
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.features = nn.Sequential(
            VGGBlock(3, 32, 2, dropout=0.1),  # 100x100
            nn.MaxPool2d(2, 2),  # 50x50
            VGGBlock(32, 64, 2, dropout=0.2),  # 50x50
            nn.MaxPool2d(2, 2),  # 25x25
            VGGBlock(64, 128, 2, dropout=0.3),  # 25x25
            nn.MaxPool2d(2, 2),  # 12x12
            VGGBlock(128, 128, 2, dropout=0.4),  # 12x12
            nn.MaxPool2d(2, 2),  # 6x6
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
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