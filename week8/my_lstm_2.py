#  -*- utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x 的形状  (batch_size, seq_length, input_size)
        out, (h_n, c_n) = self.lstm(x)  # out包含所有时间步的输出
        # 取最后一个时间步的输出
        # h_n的形状为 (num_layers, batch_size, hidden_size)
        # 取最后一层的隐藏状态
        out = self.fc(out[:, -1, :])
        return out


def run():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(
        root='../data',
        train=True,
        download=False,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 超参数设置
    input_size = 28  # 输入向量维度
    hidden_size = 128
    output_size = 10
    num_layers = 2
    learning_rate = 0.001

    # 初始化模型，损失函数和优化器
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    num_epochs = 10
    device = torch.device("mps")
    model = model.to(device)
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.view(-1, 28, 28)  # 将图片数据转换成二维张量, (batch_size, 28, 28)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item():.4f}")
    print("Training finished!")


def test():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(
        root='../data',
        train=True,
        download=False,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 超参数设置
    input_size = 28  # 输入向量维度
    hidden_size = 128
    output_size = 10
    num_layers = 2
    learning_rate = 0.001

    # 初始化模型，损失函数和优化器
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("mps")
    model = model.to(device)
    for images, labels in train_loader:
        break
    # images: torch.Size([64, 1, 28, 28])
    # labels: torch.Size([64])
    images = images.view(-1, 28, 28)  # 将图片数据转换成二维张量, (batch_size, 28, 28)
    images, labels = images.to(device), labels.to(device)
    # images: torch.Size([64, 28, 28])
    out, (h_n, c_n) = model.lstm(images)
    # out: torch.Size([64, 28, 128]) # 每个时间步的最终输出，输入维度和隐藏层一致，所以(batch_size, seq_length, hidden_size)
    # h_n: torch.Size([2, 64, 128])  # 最终隐藏状态的形状(num_layers * num_directions, batch_size, hidden_size)
    # c_n: torch.Size([2, 64, 128])  # 最终细胞状态的形状(num_layers * num_directions, batch_size, hidden_size)
    return True


if __name__ == '__main__':
    run()

    """
    result 记录
    Epoch [1 / 10], Loss: 0.1369
    Epoch [2 / 10], Loss: 0.0185
    Epoch [3 / 10], Loss: 0.0894
    Epoch [4 / 10], Loss: 0.0082
    Epoch [5 / 10], Loss: 0.0163
    Epoch [6 / 10], Loss: 0.0159
    Epoch [7 / 10], Loss: 0.0007
    Epoch [8 / 10], Loss: 0.0098
    Epoch [9 / 10], Loss: 0.0061
    Epoch [10 / 10], Loss: 0.0019
    Training finished!
    """