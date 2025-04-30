import torch
import torch.nn as nn
import torch.optim as optim

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
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

def make_layer(block, in_channels, out_channels, blocks, stride=1):
    downsample = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * block.expansion),
        )
    layers = []
    layers.append(block(in_channels, out_channels, stride, downsample))
    for _ in range(1, blocks):
        layers.append(block(out_channels * block.expansion, out_channels))
    return nn.Sequential(*layers)

class MyResNet(nn.Module):
    def __init__(self):
        super(MyResNet, self).__init__()
        self.m_name = 'ResNet18'
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 100->50
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                # 50->25

        # ResNet-18: [2, 2, 2, 2] blocks
        self.layer1 = make_layer(BasicBlock, 64, 64, 2, stride=1)                      # 25x25
        self.layer2 = make_layer(BasicBlock, 64, 128, 2, stride=2)                     # 13x13
        self.layer3 = make_layer(BasicBlock, 128, 256, 2, stride=2)                    # 7x7
        self.layer4 = make_layer(BasicBlock, 256, 512, 2, stride=2)                    # 4x4

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, 2)

        self.lr = 0.001
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        输入3x100x100
        采用ResNet18结构
        每个残差块由BasicBlock类实现
        层级为[2,2,2,2]
        多次MaxPool和步长下采样
        使用AdaptiveAvgPool2d
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
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