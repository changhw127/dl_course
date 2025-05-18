#  -*- utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义基于nn.RNN 的模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # 输入维度为(batch, seq, feature)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # RNN的输出包含输出序列和隐藏状态
        out, _ = self.rnn(x)
        # 使用最后一个时间步的输出作为分类结果
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
    model = RNNModel(input_size, hidden_size, output_size, num_layers)
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


if __name__ == '__main__':
    run()

    """
    result 记录
    Epoch 1, Loss: 0.3145672678947449
    Epoch 2, Loss: 0.09776724874973297
    Epoch 3, Loss: 0.2141588181257248
    Epoch 4, Loss: 0.3678658902645111
    Epoch 5, Loss: 0.08155558258295059
    Epoch 6, Loss: 0.012074442580342293
    Epoch 7, Loss: 0.017848746851086617
    Epoch 8, Loss: 0.059755221009254456
    Epoch 9, Loss: 0.1826942414045334
    Epoch 10, Loss: 0.09301760792732239
    Training finished!
    """