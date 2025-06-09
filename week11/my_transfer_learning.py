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


def create_model(pretrained=True, num_classes=2, feature_extract=True):
    """创建并初始化ResNet50模型"""
    model = models.resnet50(pretrained=pretrained)

    if feature_extract:
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False

    # 替换最后一层全连接层
    model.fc = nn.Linear(model.fc.in_features, 2)
    pt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model

