# -*- utf-8 -*-
import sys
import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt


class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10),
        )

        self.lr = 0.001
        self.criterion = None
        self.optimizer = None
        self.loss = None

        self.init_criterion()
        self.init_optimizer()

    def init_criterion(self):
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        return True

    def init_optimizer(self):
        # 定义优化器
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.optimizer.zero_grad()
        return True

    def forward(self, x):
        return self.model(x)

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


def progress_bar(current, total, bar_length=50):
    # 用来打印进度条
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write(f"\r进度: [{arrow + spaces}] {int(round(percent * 100))} %")
    sys.stdout.flush()

    if current == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


def log(msg, clear=False):
    # 打印并记录日志
    if clear:
        with open('log.txt', 'w') as f:
            f.close()
    with open('log.txt', 'a') as f:
        f.write(msg + '\n')
        f.close()
    print(msg)
    return True


def get_device(use_gpu=True):
    # 选择是否使用GPU加速
    if torch.backends.mps.is_available() and use_gpu:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log(f"Using device: {device}")
    return device


def prepare_data(batch_size=64, is_subset=False, subset_percent=0.01):
    # MNIST数据加载,截取数据，返回DataLoader, 支持获取子集用于跑通模型
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST(root='../data', train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root='../data', train=False, download=False, transform=transform)
    # torch.Size([60000, 28, 28])
    # torch.Size([10000, 28, 28])

    # 截取数据用于跑通模型
    if is_subset:
        # 默认获取 1% 的数据用于拉通
        indices = torch.randperm(len(train_data))[:int(len(train_data) * subset_percent)]
        train_data = Subset(train_data, indices)
        indices = torch.randperm(len(test_data))[:int(len(test_data) * subset_percent)]
        test_data = Subset(test_data, indices)
        # len(train_data.indices) 600
        # len(test_data.indices)  100

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    log(f"train_data: {len(train_loader.dataset)}, test_data: {len(test_loader.dataset)}, batch_size: {batch_size}")
    return train_data, test_data, train_loader, test_loader



def train_model(model, device, data_loader, epoch):
    model.train()
    loss = 0
    correct = 0

    ix = 0
    total = len(data_loader)
    for batch, (X, Y) in enumerate(data_loader):
        ix += 1
        X, Y = X.to(device), Y.to(device)
        # 4步
        y = model.forward(X)
        model.calc_loss(y, Y, save_loss=True)
        model.backward()
        model.update_params()

        loss += model.loss.item()
        pred = y.argmax(dim=1, keepdim=True)
        correct += pred.eq(Y.view_as(pred)).sum().item()
        progress_bar(ix, total)


    loss /= len(train_loader)
    acc = 100. * correct / len(data_loader.dataset)
    log(f"Epoch {epoch + 1}, Train Loss: {loss:.4f}, Accuracy: {acc:.4f} %")
    return loss, acc


def test_model(model, device, data_loader):
    model.eval()
    loss = 0
    correct = 0
    # 关闭梯度计算，用于加速
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            y = model.forward(X)
            loss += model.calc_loss(y, Y, save_loss=False)
            pred = y.argmax(dim=1, keepdim=True)
            correct += pred.eq(Y.view_as(pred)).sum().item()

    loss /= len(data_loader)
    acc = 100. * correct / len(data_loader.dataset)
    log(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f} %")
    return loss, acc


def plt_result(train_loss_ls, train_acc_ls, test_loss_ls, test_acc_ls):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    train_loss = [t.cpu().item() if torch.is_tensor(t) else t for t in train_loss_ls]
    test_loss = [t.cpu().item() if torch.is_tensor(t) else t for t in test_loss_ls]

    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_ls, label='Train Accuracy')
    plt.plot(test_acc_ls, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    # plt.show()
    plt.savefig('my_lenet.png', dpi=300)
    log('result plt ok.')
    return True



if __name__ == '__main__':
    train_loss_ls, train_acc_ls = [], []
    test_loss_ls, test_acc_ls = [], []
    epochs = 10
    batch_size = 64
    st = datetime.datetime.now()
    log(f'start ==========================\nepoch {epochs}', clear=True)
    device = get_device(use_gpu=True)
    model = MyLeNet().to(device)

    # train_data, test_data, train_loader, test_loader = prepare_data(batch_size=batch_size, is_subset=True, subset_percent=0.05)
    train_data, test_data, train_loader, test_loader = prepare_data(is_subset=False)
    for epoch in range(epochs):
        train_loss, train_acc = train_model(model, device, train_loader, epoch)
        test_loss, test_acc = test_model(model, device, test_loader)

        train_loss_ls.append(train_loss)
        train_acc_ls.append(train_acc)
        test_loss_ls.append(test_loss)
        test_acc_ls.append(test_acc)

    plt_result(train_loss_ls, train_acc_ls, test_loss_ls, test_acc_ls)

    ed = datetime.datetime.now()
    log(f'cost {(ed - st).total_seconds()} s')
    log(f'最终结果: 测试损失: {test_loss_ls[-1]:.4f}, 准确率: {test_acc_ls[-1]:.4f}')
    log('finished.')
