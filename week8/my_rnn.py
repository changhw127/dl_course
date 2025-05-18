#  -*- utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size

        # 定义权重矩阵
        self.Wxa = nn.Parameter(torch.randn(input_size, hidden_size))  # torch.Size([28, 128]) 输入到隐藏层的权重矩阵
        self.Waa = nn.Parameter(torch.randn(hidden_size, hidden_size))  # torch.Size([128, 128])  隐藏层到隐藏层的权重矩阵
        self.Wy = nn.Parameter(torch.randn(hidden_size, output_size))  # torch.Size([128, 10]) 隐藏层到输出层的权重矩阵

        # 定义偏置
        self.ba = nn.Parameter(torch.zeros(hidden_size))  # torch.Size([128]) 隐藏层的偏置
        self.by = nn.Parameter(torch.zeros(output_size))  # torch.Size([10]) 输出层的偏置

    def forward(self, x):
        # 初始化隐藏层状态
        a_t = torch.zeros(x.size()[0], self.hidden_size)  # torch.Size([64, 128])

        # rnn的计算过程
        for t in range(x.size()[1]):  # 遍历时间步
            x_t = x[:, t, :]  # 获取当前时间步的输入 torch.Size([64, 28])
            """
            x_t: torch.Size([28, 128])
            Wxa: torch.Size([28, 128])
            a_t: torch.Size([64, 128])
            Waa: torch.Size([128, 128])
            ba: torch.Size([128])
            
            torch.mm(x_t, self.Wxa): torch.Size([64, 128]) 
            torch.mm(a_t, self.Waa): torch.Size([64, 128])
            a_t_new: torch.Size([64, 128])
            """
            a_t = torch.tanh(torch.mm(x_t, self.Wxa) + torch.mm(a_t, self.Waa) + self.ba)  # 计算隐藏层状态
        # 输出层计算
        y = torch.mm(a_t, self.Wy) + self.by
        return y


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
    learning_rate = 0.001

    # 初始化模型，损失函数和优化器
    model = CustomRNN(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images = images.view(-1, 28, 28)  # 将图片数据转换成二维张量, (batch_size, 28, 28)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    print("Training finished!")


def test():
    # 假设 batch_size=64, input_size=28, hidden_size=128
    x_t = torch.randn(64, 28)  # (64, 28)
    h_t = torch.randn(64, 128)  # (64, 128)
    Wxa = torch.randn(28, 128)  # (28, 128)
    Waa = torch.randn(128, 128)  # (128, 128)
    ba = torch.randn(128)  # (128,)

    # 计算过程
    term1 = torch.mm(x_t, Wxa)  # (64, 28) @ (28, 128) → (64, 128)
    term2 = torch.mm(h_t, Waa)  # (64, 128) @ (128, 128) → (64, 128)
    term3 = ba  # (128,) → 广播为 (64, 128)

    output = term1 + term2 + term3  # 所有形状 (64, 128)


if __name__ == '__main__':
    run()

    """
    result 记录
    Epoch 1, Loss: 9.376683235168457
    Epoch 2, Loss: 5.6001081466674805
    Epoch 3, Loss: 5.141937255859375
    Epoch 4, Loss: 3.472097635269165
    Epoch 5, Loss: 2.9716637134552
    Epoch 6, Loss: 2.1581902503967285
    Epoch 7, Loss: 2.734004020690918
    Epoch 8, Loss: 2.162174940109253
    Epoch 9, Loss: 2.1528894901275635
    Epoch 10, Loss: 2.223137617111206
    Training finished!
    """