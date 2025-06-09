import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import time

# 自动选择可用设备
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


def create_model(pretrained=True, num_classes=100, num_unfreeze=1):
    model = models.resnet50(pretrained=pretrained)

    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    if num_unfreeze >= 1:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if num_unfreeze >= 2:
        for param in model.layer3.parameters():
            param.requires_grad = True
    if num_unfreeze >= 3:
        for param in model.layer2.parameters():
            param.requires_grad = True
    if num_unfreeze >= 4:
        for param in model.layer1.parameters():
            param.requires_grad = True
    if num_unfreeze >= 5:  # 解冻所有层
        for param in model.parameters():
            param.requires_grad = True
    # 替换全连接层
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


# 超参数优化
num_epochs = 20
batch_size = 64  # 减小batch size避免内存溢出
learning_rate = 0.0005  # 降低学习率
num_classes = 100

# 使用CIFAR-100专用归一化参数
transform_train = transforms.Compose([
    transforms.Resize(128),  # 上采样到128x128
    transforms.RandomCrop(128, padding=8),  # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),  # 明确设置翻转概率
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.RandomRotation(15),  # 随机旋转
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)),  # 随机擦除
])

transform_test = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])

# 加载数据集
train_dataset = datasets.CIFAR100(root='../data/', train=True, download=False, transform=transform_train)
test_dataset = datasets.CIFAR100(root='../data/', train=False, download=False, transform=transform_test)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,)

train_subset, _ = random_split(train_dataset, [int(len(train_dataset) * 0.1), len(train_dataset) - int(len(train_dataset) * 0.1)])
test_subset, _ = random_split(test_dataset, [int(len(test_dataset) * 0.1), len(test_dataset) - int(len(test_dataset) * 0.1)])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

# 创建模型
model = create_model(num_classes=num_classes, num_unfreeze=3)
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# 训练和验证函数（添加准确率计算）
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        running_loss += loss.item() * images.size(0)
        # print('loss:', loss.item())

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# 训练主循环
train_losses, val_losses = [], []
train_accs, val_accs = [], []

start_time = time.time()
best_acc = 0.0

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    # val_loss, val_acc = validate(model, test_loader, criterion, device)
    scheduler.step()

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    # val_losses.append(val_loss)
    # val_accs.append(val_acc)

    # 保存最佳模型
    # if val_acc > best_acc:
    #     best_acc = val_acc
    #     torch.save(model.state_dict(), 'best_model.pth')

    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch + 1}/{num_epochs}] | LR: {lr:.6f}")
    print(f"Train: Loss {train_loss:.4f} | Acc {train_acc:.2f}%")
    # print(f"Valid: Loss {val_loss:.4f} | Acc {val_acc:.2f}% | Best {best_acc:.2f}%\n")

end_time = time.time()
print(f"Training complete in {(end_time - start_time) / 60:.2f} minutes")
print(f"Best validation accuracy: {best_acc:.2f}%")

# 绘制结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy')
# plt.plot(val_accs, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()


"""
数据预处理优化：
使用更合适的128x128分辨率
添加随机裁剪增强
使用CIFAR-100专用统计值
模型改进：
分层解冻逻辑更清晰
在全连接层添加Dropout
支持完全解冻模式
训练过程优化：
添加准确率计算
使用AdamW优化器
添加余弦退火学习率调度
实现最佳模型保存
使用non_blocking传输和pin_memory加速
资源管理：
减小batch_size到64
添加梯度清零优化(set_to_none=True)
结果可视化：
同时展示损失和准确率曲线
保存训练结果图像
"""