#  -*- utf-8 -*-
import torch
import torchvision.models as models
import torch.nn as nn


def sample():
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(models.fc.in_features, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pass
