#  -*- utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CustomLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # 权重初始化
        self.Wf = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))  # 遗忘 gate
        self.Wi = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))  # 输入 gate
        self.Wc = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))  # 候选记忆 gate
        self.Wo = nn.Parameter(torch.randn(hidden_size + input_size, hidden_size))  # 输出 gate

        # 全连接层，输出维度为 output_size，分类结果
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev, c_prev):
        """
        :param x: 当前时刻的输入 (batch_size, input_size)
        :param h_prev: 前一时刻的隐藏状态 (batch_size, hidden_size)
        :param c_prev: 前一时刻的细胞状态 (batch_size, hidden_size)
        :return: 当前时刻的隐藏状态，当前时刻的细胞状态
        """

        # 拼接输入和前一时刻的隐藏状态作为当前时刻的输入
        combined = torch.cat((x, h_prev), dim=1)  # (batch_size, hidden_size + input_size)

        # 计算遗忘 gate、输入 gate、候选记忆 gate、输出 gate
        f_t = torch.sigmoid(torch.mm(combined, self.Wf))  # 遗忘 gate
        i_t = torch.sigmoid(torch.mm(combined, self.Wi))  # 输入 gate
        c_tilda_t = torch.tanh(torch.mm(combined, self.Wc))  # 候选记忆 gate
        o_t = torch.sigmoid(torch.mm(combined, self.Wo))  # 输出 gate

        # 更新细胞状态
        c_t = f_t * c_prev + i_t * c_tilda_t

        # 更新隐藏状态
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

    def init_hidden(self, batch_size):
        """
        初始化隐藏状态和细胞状态
        :param batch_size: 批量大小
        :return: 隐藏状态和细胞状态
        """
        h0 = torch.zeros(batch_size, self.hidden_size)
        c0 = torch.zeros(batch_size, self.hidden_size)
        return h0, c0


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
    model = CustomLSTM(input_size, hidden_size, output_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.view(-1, 28, 28)  # 将图片数据转换成二维张量, (batch_size, 28, 28)
            batch_size = images.size(0)
            # 初始化隐藏状态和细胞状态
            h_t, c_t = model.init_hidden(batch_size)

            # 前向传播
            for t in range(images.size(1)):  # 遍历时间序列
                h_t, c_t = model(images[:, t, :], h_t, c_t)

            # 通过最后一个时间步的隐藏状态进行分类
            outputs = model.fc(h_t)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item():.4f}")
    print("Training finished!")


if __name__ == '__main__':
    run()

    """
    result 记录
    Epoch [1 / 10], Loss: 2.1450
    Epoch [2 / 10], Loss: 1.9761
    Epoch [3 / 10], Loss: 1.7638
    Epoch [4 / 10], Loss: 1.9560
    Epoch [5 / 10], Loss: 1.9926
    Epoch [6 / 10], Loss: 1.6968
    Epoch [7 / 10], Loss: 1.8951
    Epoch [8 / 10], Loss: 1.7768
    Epoch [9 / 10], Loss: 1.9073
    Epoch [10 / 10], Loss: 1.7011
    Training finished!
    """
