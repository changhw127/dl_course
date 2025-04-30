# -*- utf-8 -*-
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PATH = 'data/course/'

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)  # 批量规范化
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1x1conv:
            # 使用1x1，使得输出通道数匹配
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)


class ResNetBlock(object):
    def __init__(self, input_channels, num_channels, num_residuals, first_block=False):
        self.blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                self.blk.append(Residual(num_channels, num_channels))


class MyResNet(object):
    def __init__(self):
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.b2 = nn.Sequential(*ResNetBlock(64, 64, 2, first_block=True).blk)
        self.b3 = nn.Sequential(*ResNetBlock(64, 128, 2).blk)
        self.b4 = nn.Sequential(*ResNetBlock(128, 256, 2).blk)
        self.b5 = nn.Sequential(*ResNetBlock(256, 512, 2).blk)
        self.net = nn.Sequential(
            self.b1,  self.b2,  self.b3,  self.b4,  self.b5,
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )


def test():
    blk = Residual(3, 3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)
    # torch.Size([4, 3, 6, 6])

    blk = Residual(3, 6, use_1x1conv=True, strides=2)
    blk(X).shape
    #  torch.Size([4, 6, 3, 3])

    x = Image.open(PATH + 'cat.jpg')
    x = np.array(x)
    x = x[::2, ::2]
    x = x[:224, :224]
    img = x.reshape(1, 3, 224, 224)
    img = torch.tensor(img, dtype=torch.float32)

    MyResNet().net(img)
    return