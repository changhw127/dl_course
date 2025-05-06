
import torch.nn as nn
import torch.optim as optim

class MyVGG(nn.Module):
    def __init__(self,  lr=0.001, opt='Adam', num_classes=62):
        super(MyVGG, self).__init__()
        self.m_name = 'small_VGG'
        self.opt = opt
        self.lr = lr
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 30x30 -> 30x30
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 30x30 -> 30x30
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 30x30 -> 15x15

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 15x15 -> 15x15
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 15x15 -> 15x15
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 15x15 -> 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 7x7 -> 7x7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 7x7 -> 3x3
        )
        self.plt_features_indexes = None
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*3*3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.init_criterion()
        self.init_optimizer()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_desc(self):
        desc = '''
        输入30x30。类别62,small vgg
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

class MyVGG1(nn.Module):
    def __init__(self,  lr=0.001, opt='Adam', num_classes=62):
        super(MyVGG1, self).__init__()
        self.m_name = 'small_VGG'
        self.opt = opt
        self.lr = lr
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 30x30 -> 30x30
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), # 30x30 -> 30x30
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 30x30 -> 15x15

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 15x15 -> 15x15
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # 15x15 -> 15x15
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 15x15 -> 7x7

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 7x7 -> 7x7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                             # 7x7 -> 3x3
        )
        self.plt_features_indexes = None
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*3*3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.init_criterion()
        self.init_optimizer()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_desc(self):
        desc = '''
        输入30x30。类别62,small vgg
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