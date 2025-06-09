#  -*- utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import numpy as np
import os
import time
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torchvision.utils import make_grid
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
matplotlib.rcParams['font.family'] = 'Heiti TC'  # 可以替换为其他字体

# 设置随机种子确保可复现性
# torch.manual_seed(42)
# np.random.seed(42)
# torch.manual_seed(4)
# np.random.seed(4)

# 检查GPU可用性
device = torch.device('mps')
print(f"使用设备: {device}")

# 数据集路径设置
data_dir = '../data/dog_cat/train_sub'  # 包含cat和dog子目录的文件夹

# 数据增强和归一化
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 创建数据集
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir), data_transforms[x])
                  for x in ['train', 'val']}

# 手动划分训练集和验证集 (80%训练, 20%验证)
dataset_size = len(image_datasets['train'])
indices = list(range(dataset_size))
split = int(np.floor(0.2 * dataset_size))
np.random.shuffle(indices)

train_indices, val_indices = indices[split:], indices[:split]

# 创建数据加载器
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

batch_size = 32
num_workers = 4 if torch.cuda.is_available() else 0

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                         sampler=train_sampler, num_workers=num_workers),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                       sampler=val_sampler, num_workers=num_workers)
}

dataset_sizes = {'train': len(train_indices), 'val': len(val_indices)}
class_names = image_datasets['train'].classes

print(f"类别: {class_names}")
print(f"训练集大小: {dataset_sizes['train']}")
print(f"验证集大小: {dataset_sizes['val']}")


# 可视化一些样本
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 暂停一下以便更新绘图


# 获取一批训练数据
inputs, classes = next(iter(dataloaders['train']))

# 创建网格显示
out = make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])


# 创建模型
def create_model(pretrained=True, num_classes=2, feature_extract=True, num_unfreeze=0):
    model = models.resnet50(pretrained=pretrained)
    if feature_extract:
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

    # 解冻指定层数
    layers_to_unfreeze = {
        1: [model.layer4],
        2: [model.layer3, model.layer4],
        3: [model.layer2, model.layer3, model.layer4],
        4: [model.layer1, model.layer2, model.layer3, model.layer4]
    }

    if num_unfreeze > 0:
        for layer in layers_to_unfreeze[num_unfreeze]:
            for param in layer.parameters():
                param.requires_grad = True
    # 替换最后一层全连接层
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


model = create_model(num_unfreeze=2)
model = model.to(device)

# 设置训练参数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 学习率调度器
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# 训练函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # 记录训练历史
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs - 1 + 1}')
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()  # 设置模型为评估模式
            running_loss = 0.0
            running_corrects = 0
            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # 梯度清零
                optimizer.zero_grad()
                # 前向传播
                # 只在训练时跟踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # 只在训练时反向传播 + 优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'训练完成于 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证准确率: {best_acc:.4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history


# 训练模型
num_epochs = 15
model, history = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)


# 可视化训练过程
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='训练集')
    plt.plot(history['val_acc'], label='验证集')
    plt.title('模型准确率')
    plt.ylabel('准确率')
    plt.xlabel('Epoch')
    plt.legend()

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='训练集')
    plt.plot(history['val_loss'], label='验证集')
    plt.title('模型损失')
    plt.ylabel('损失')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_history(history)


# 模型评估
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds


val_labels, val_preds = evaluate_model(model, dataloaders['val'])


# 混淆矩阵
def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()


# 分类报告
print("\n分类报告:")
print(classification_report(val_labels, val_preds, target_names=class_names))

plot_confusion_matrix(val_labels, val_preds, class_names)


# 可视化预测结果
def visualize_predictions(model, dataloader, num_images=6):
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(12, 8))

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            ax.set_title(f'预测: {class_names[preds[j]]}\n真实: {class_names[labels[j]]}')
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


# 可视化一些验证集预测
# visualize_predictions(model, dataloaders['val'])

# 保存模型
torch.save(model.state_dict(), 'cats_dogs_resnet50.pth')
print("模型已保存为 'cats_dogs_resnet50.pth'")


# 微调模型（可选进阶步骤）
def fine_tune_model(model):
    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True

    # 使用更小的学习率
    optimizer_ft = optim.Adam(model.parameters(), lr=0.0001)

    # 学习率调度器
    exp_lr_scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    return model, optimizer_ft, exp_lr_scheduler_ft

# 微调模型（取消注释以执行）
# model, optimizer_ft, exp_lr_scheduler_ft = fine_tune_model(model)
# model, history_ft = train_model(model, criterion, optimizer_ft, exp_lr_scheduler_ft, num_epochs=10)



"""
Epoch 0/14
----------
train Loss: 0.3888 Acc: 0.8022
val Loss: 0.1128 Acc: 0.9740
Epoch 1/14
----------
train Loss: 0.2379 Acc: 0.8976
val Loss: 0.1142 Acc: 0.9600
Epoch 2/14
----------
train Loss: 0.2098 Acc: 0.9081
val Loss: 0.0668 Acc: 0.9800
Epoch 3/14
----------
train Loss: 0.2358 Acc: 0.8946
val Loss: 0.0994 Acc: 0.9580
Epoch 4/14
----------
train Loss: 0.1976 Acc: 0.9101
val Loss: 0.0818 Acc: 0.9660
Epoch 5/14
----------
train Loss: 0.2153 Acc: 0.9021
val Loss: 0.0609 Acc: 0.9800
Epoch 6/14
----------
train Loss: 0.2128 Acc: 0.9036
val Loss: 0.0567 Acc: 0.9800
Epoch 7/14
----------
train Loss: 0.1832 Acc: 0.9251
val Loss: 0.0525 Acc: 0.9820
Epoch 8/14
----------
train Loss: 0.1806 Acc: 0.9116
val Loss: 0.0516 Acc: 0.9820
Epoch 9/14
----------
train Loss: 0.1618 Acc: 0.9316
val Loss: 0.0515 Acc: 0.9840
Epoch 10/14
----------
train Loss: 0.1670 Acc: 0.9251
val Loss: 0.0517 Acc: 0.9840
Epoch 11/14
----------
train Loss: 0.1750 Acc: 0.9231
val Loss: 0.0489 Acc: 0.9820
Epoch 12/14
----------
train Loss: 0.1560 Acc: 0.9351
val Loss: 0.0517 Acc: 0.9860
Epoch 13/14
----------
train Loss: 0.1837 Acc: 0.9181
val Loss: 0.0493 Acc: 0.9820
Epoch 14/14
----------
train Loss: 0.1622 Acc: 0.9301
val Loss: 0.0489 Acc: 0.9820
训练完成于 6分 1秒
最佳验证准确率: 0.9860
分类报告:
              precision    recall  f1-score   support

        cats       0.99      0.98      0.99       250
        dogs       0.98      0.99      0.99       250

    accuracy                           0.99       500
   macro avg       0.99      0.99      0.99       500
weighted avg       0.99      0.99      0.99       500


解冻层数	训练时间	内存使用	验证准确率	过拟合风险	适用数据集大小
0（全冻结）	★☆☆☆☆（最快）	★☆☆☆☆（最低）	★★☆☆☆（97-98%）	★☆☆☆☆（最低）	<1,000样本
1（顶层）	★★☆☆☆	★★☆☆☆	★★★☆☆（98-98.5%）	★★☆☆☆	1,000-5,000样本
2（上中层）	★★★☆☆	★★★☆☆	★★★★☆（98.5-99%）	★★★☆☆	5,000-20,000样本
3（中层）	★★★★☆	★★★★☆	★★★★★（99-99.2%）	★★★★☆	20,000-50,000样本
4（全解冻）	★★★★★（最慢）	★★★★★（最高）	★★★★★（99.2-99.5%）	★★★★★（最高）	>50,000样本
"""