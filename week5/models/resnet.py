# -*- utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

class BasicBlock(nn.Module):
    expansion = 1 # 输出通道数相对于 out_channels 的扩展倍数
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        # in_channels：输入特征图的通道数。
        # out_channels：卷积层输出的通道数。
        # stride：第一层卷积的步长，默认为1，用于控制特征图尺寸变化。
        # downsample：下采样函数（通常是一个卷积+BN），用于调整残差连接的尺寸和通道数，使其与主路径输出匹配。
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class MyResNet(nn.Module):
    def __init__(self, lr=0.01, opt='SGD'):
        super(MyResNet, self).__init__()
        self.m_name = 'ResNet_Small'
        self.raw_size = 100 # 原始224
        self.lr = lr
        self.opt = opt
        self.criterion = None
        self.optimizer = None
        self.loss = None

        # 通道数适当减小，适合100x100输入
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 100x100
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # 每个layer包含2个BasicBlock
        self.layer1 = self._make_layer(32, 2, stride=1)   # 100x100
        self.layer2 = self._make_layer(64, 2, stride=2)   # 50x50
        self.layer3 = self._make_layer(128, 2, stride=2)  # 25x25
        self.layer4 = self._make_layer(256, 2, stride=2)  # 13x13

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)

        self.init_criterion()
        self.init_optimizer()

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def get_desc(self):
        desc = '''
        ResNet简化版，适合100x100彩色图像二分类
        通道数32-64-128-256，残差块数较少
        每层2个BasicBlock
        最后全局平均池化+全连接
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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