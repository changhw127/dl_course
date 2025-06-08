#  -*- utf-8 -*-
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
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
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    batch_size = 128
    lr = 0.0002
    epochs = 10
    latent_dim = 100
    dataloader = DataLoader(
        datasets.MNIST('../data', train=True, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),),
        batch_size=batch_size, shuffle=True
    )
    G = Generator(latent_dim)
    D = Discriminator()

    optim_G = optim.Adam(G.parameters(), lr=lr)
    optim_D = optim.Adam(D.parameters(), lr=lr)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for idx, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)

            real_label = torch.ones((batch_size, 1))
            fake_label = torch.zeros((batch_size, 1))

            real_images = real_images.view(batch_size, -1)

            # 训练判别器
            optim_D.zero_grad()
            real_output = D(real_images)
            loss_real = criterion(real_output, real_label)

            z = torch.randn(batch_size, latent_dim)
            fake_images = G(z).detach()
            fake_output = D(fake_images)
            loss_fake = criterion(fake_output, fake_label)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optim_G.step()

            # 训练生成器
            optim_G.zero_grad()

            z = torch.randn(batch_size, latent_dim)
            gen_images = G(z)

            output = D(gen_images)
            loss_G = criterion(output, real_label)
            loss_G.backward()
            optim_G.step()

            if idx % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], Step [{idx}/{len(dataloader)}], '
                      f'Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}')


"""
Epoch [9/10], Step [400/469], Loss_D: 3.0398, Loss_G: 0.0953
"""



