import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.m_name = 'AlexNet'
        self.num_classes = 2
        self.lr = 0.001
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.features = nn.Sequential(
            # 第一层卷积: 100x100 -> 24x24
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 24x24 -> 11x11

            # 第二层卷积: 11x11 -> 11x11
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 11x11 -> 5x5

            # 第三层卷积: 5x5 -> 5x5
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第四层卷积: 5x5 -> 5x5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 5x5 -> 2x2

            # 第五层卷积: 5x5 -> 5x5
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),  # 5x5 -> 2x2
        )

        # 移除AdaptiveAvgPool2d，直接使用Flatten ？
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 128),  # 2x2的特征图
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, self.num_classes),
        )

        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        输入3x100x100

        去除一层卷积
        使用AdaptiveAvgPool2d
        '''.replace('        ', '')
        return desc

    def init_criterion(self):
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        return True

    def init_optimizer(self):
        # 定义优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.optimizer.zero_grad()
        return True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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


class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.m_name = 'AlexNet'
        self.num_classes = 2
        self.lr = 0.001
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.features = nn.Sequential(
            # 第一层卷积: 100x100 -> 24x24
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 24x24 -> 11x11

            # 第二层卷积: 11x11 -> 11x11
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 11x11 -> 5x5

            # 第三层卷积: 5x5 -> 5x5
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 第四层卷积: 5x5 -> 5x5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 5x5 -> 2x2

            # 第五层卷积: 5x5 -> 5x5
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),  # 5x5 -> 2x2
        )

        # 移除AdaptiveAvgPool2d，直接使用Flatten ？
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, 128),  # 2x2的特征图
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, self.num_classes),
        )

        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        输入3x100x100
        
        去除一层卷积
        使用AdaptiveAvgPool2d
        '''.replace('        ', '')
        return desc

    def init_criterion(self):
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        return True

    def init_optimizer(self):
        # 定义优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.optimizer.zero_grad()
        return True

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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


class SimpleAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 12 * 12, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)