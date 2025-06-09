#  -*- utf-8 -*-
import random

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


class Generator1(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),


            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    batch_size = 64
    lr = 0.0002
    epochs = 10
    latent_dim = 100
    device = torch.device('mps')
    dataloader = DataLoader(
        datasets.MNIST('../data', train=True, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),),
        batch_size=batch_size, shuffle=True
    )
    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)

    optim_G = optim.Adam(G.parameters(), lr=lr)
    optim_D = optim.Adam(D.parameters(), lr=lr * 0.8)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for idx, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)

            # real_label = torch.ones((batch_size, 1))
            # fake_label = torch.zeros((batch_size, 1))
            real_label = torch.full((batch_size, 1), 0.9, device=device)  # 代替1.0
            fake_label = torch.full((batch_size, 1), 0.1, device=device)  # 代替0.0

            real_images = real_images.view(batch_size, -1).to(device)

            # 训练判别器
            optim_D.zero_grad()
            real_output = D(real_images)
            loss_real = criterion(real_output, real_label)

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = G(z).detach()
            fake_output = D(fake_images)
            loss_fake = criterion(fake_output, fake_label)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_D.step()

            # 训练生成器
            for _ in range(random.randint(1,2)):
                optim_G.zero_grad()

                z = torch.randn(batch_size, latent_dim).to(device)
                gen_images = G(z)

                output = D(gen_images)
                loss_G = criterion(output, real_label)
                loss_G.backward()
                optim_G.step()

            if idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{idx}/{len(dataloader)}], '
                      f'Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')
        # print(f'Epoch [{epoch + 1}/{epochs}], Step [{idx}/{len(dataloader)}], '
        #       f'Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')


"""
Epoch [10/10], Step [400/469], Loss_D: 0.6713, Loss_G: 2.2047
调整：
lr区分
标签平滑
生成器模型架构改变
1:k的训练轮次
batch调整

Epoch [10/10], Step [900/938], Loss_D: 1.4485, Loss_G: 0.6582
D的训练轮次和lr进行调整


Epoch [10/10], Step [400/469], Loss_D: 1.1687, Loss_G: 0.9292

Epoch [10/10], Step [900/938], Loss_D: 1.3604, Loss_G: 0.7878

建议
1. **调整学习率**：可以尝试降低判别器的学习率或者提高生成器的学习率，使得生成器能够更好地学习。也可以使用两个优化器不同的学习率。
2. **标签平滑（Label Smoothing）**：在真实数据的标签上使用0.9（而不是1.0），在生成数据的标签上使用0.1（而不是0.0），这样可以让判别器不要过度自信，从而给生成器更多的学习机会。
3. **生成器结构优化**：当前生成器是全连接层，可以考虑使用卷积生成器（如DCGAN结构），这通常能生成质量更高的图像。
4. **梯度惩罚（Gradient Penalty）**：考虑使用WGAN-GP，它通过梯度惩罚来稳定训练，避免模式崩溃和训练不稳定。
5. **增加生成器的能力**：可以尝试增加生成器的层数或每层的神经元数量，使其更强大。
6. **调整训练频率**：通常的做法是训练判别器多次，再训练生成器一次（例如5:1），但这里判别器已经很强，可以反过来，训练生成器多次，判别器一次（例如1:5）？但更常见的做法是让判别器多训练几次。实际上，从Loss来看，判别器已经很强，所以可以尝试减少判别器的训练次数（即让生成器多训练几次）。但注意，在代码中目前是各训练一次，可以尝试每训练生成器k次，再训练判别器一次（k>1）。
7. **使用不同的损失函数**：将生成器的损失函数改为`-log(D(G(z)))`（即非饱和损失），虽然代码中生成器使用的是最小化真实标签的交叉熵（即`criterion(output, real_label)`），这实际上就是非饱和损失的一种形式（因为生成器希望判别器将生成样本判为1）。但是，生成器Loss为2.08，说明生成样本被判别器判为假（即0）的概率很高，所以生成器损失大。另一种选择是使用Wasserstein损失。
8. **检查归一化**：确保数据归一化到[-1,1]之间（前面已经提到过），因为生成器的最后一层是Tanh，输出范围是[-1,1]。
9. **使用批归一化（BatchNorm）或层归一化（LayerNorm）**：在生成器和判别器中加入批归一化层，这有助于稳定训练。
10. **调整激活函数**：在生成器中，除了最后一层使用Tanh，其他层可以使用ReLU。判别器中使用LeakyReLU是好的。
11. **增加训练轮数**：10个epoch可能不够，可以尝试增加epoch。
12. **加入Dropout**：在判别器的某些层加入Dropout，防止过拟合。
"""



