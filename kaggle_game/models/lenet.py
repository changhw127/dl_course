# -*- utf-8 -*-
import torch.nn as nn
import torch.optim as optim


class MyLeNet(nn.Module):
    def __init__(self, lr=0.001, opt='Adam'):
        super(MyLeNet, self).__init__()
        self.m_name = 'LeNet'
        self.opt = opt
        self.lr = lr
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.features = nn.Sequential(
            # 30 x 30
            nn.Conv2d(3, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.plt_features_indexes = None
        self.classifier = nn.Sequential(
            # 全局平均池化 + 缩1层
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.Dropout(p=0.2),
            nn.Linear(128, 62),
        )
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256 * 3 * 3, 512),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 512),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 62),
        # )

        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        输入30x30。类别62
        使用BN
        使用AdaptiveAvgPool2d
        使用Dropout
        '''.replace('        ', '')
        return desc

    def init_criterion(self):
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        return True

    def init_optimizer(self):
        # 定义优化器
        if self.opt == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        if self.opt == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=1e-4)
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

    def get_plot_feature_hooks(self):
        if self.plt_features_indexes is None:
            return self.features
        else:
            feats = []
            for ix in range(len(self.features)):
                if ix in self.plt_features_indexes:
                    feats.append(self.features[ix])
            return feats

class MyLeNet2(nn.Module):
    def __init__(self, lr=0.001, opt='Adam'):
        super(MyLeNet2, self).__init__()
        self.m_name = 'LeNet'
        self.opt = opt
        self.lr = lr
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.features = nn.Sequential(
            # 30 x 30
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.plt_features_indexes = None
        self.classifier = nn.Sequential(
            # 全局平均池化 + 缩1层
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 62),
        )
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256 * 3 * 3, 512),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 512),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 62),
        # )

        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        输入30x30。类别62
        使用BN
        使用AdaptiveAvgPool2d
        使用Dropout
        '''.replace('        ', '')
        return desc

    def init_criterion(self):
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        return True

    def init_optimizer(self):
        # 定义优化器
        if self.opt == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        if self.opt == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=1e-4)
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

    def get_plot_feature_hooks(self):
        if self.plt_features_indexes is None:
            return self.features
        else:
            feats = []
            for ix in range(len(self.features)):
                if ix in self.plt_features_indexes:
                    feats.append(self.features[ix])
            return feats

class MyLeNet1(nn.Module):
    def __init__(self, lr=0.001, opt='Adam'):
        super(MyLeNet1, self).__init__()
        self.m_name = 'LeNet'
        self.opt = opt
        self.lr = lr
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.features = nn.Sequential(
            # 30 x 30
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.plt_features_indexes = None
        self.classifier = nn.Sequential(
            # 全局平均池化 + 缩1层
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.Dropout(p=0.2),
            nn.Linear(256, 62),
        )
        # self.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(256 * 3 * 3, 512),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 512),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(512, 62),
        # )

        self.init_criterion()
        self.init_optimizer()

    def get_desc(self):
        desc = '''
        输入30x30。类别62
        使用BN
        使用AdaptiveAvgPool2d
        使用Dropout
        '''.replace('        ', '')
        return desc

    def init_criterion(self):
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        return True

    def init_optimizer(self):
        # 定义优化器
        if self.opt == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        if self.opt == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=1e-4)
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

    def get_plot_feature_hooks(self):
        if self.plt_features_indexes is None:
            return self.features
        else:
            feats = []
            for ix in range(len(self.features)):
                if ix in self.plt_features_indexes:
                    feats.append(self.features[ix])
            return feats

