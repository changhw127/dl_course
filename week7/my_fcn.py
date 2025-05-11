#  -*- utf-8 -*-
import os.path

from PIL import Image
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from utlis.globals import Logger, plt_loss_result, get_device, save_model, load_model
import matplotlib.pyplot as plt
from typing import Optional


class VOCSegmentationDataset(Dataset):
    def __init__(self, root, image_set='train', transform=None):
        self.root = root
        self.transform = transform or self._transform()

        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')
        split_file = os.path.join(root, 'ImageSets/Segmentation', f'{image_set}.txt')

        with open(split_file, 'r') as f:
            file_names = f.read().splitlines()

        self.images = [os.path.join(image_dir, f'{x}.jpg') for x in file_names]
        self.masks = [os.path.join(mask_dir, f'{x}.png') for x in file_names]

    def _transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        image = image.resize((256, 256), Image.BILINEAR)
        mask = mask.resize((256, 256), Image.NEAREST)

        if self.transform:
            image = self.transform(image)
            mask = np.array(mask, dtype=np.int64)

            mask[mask > 20] = 255
            mask = torch.from_numpy(mask)

        return image, mask

    @classmethod
    def collate_fn(cls, batch):
        images, masks = zip(*batch)
        images = torch.stack(images)
        masks = torch.stack(masks)
        return images, masks

class FCN(nn.Module):
    """
    定义FCN网络，使用ResNET18作为编码器
    """
    def __init__(self, num_class, lr=0.001, opt='Adam'):
        super(FCN, self).__init__()

        self.lr = lr
        self.opt = opt
        self.criterion = None
        self.optimizer = None
        self.loss = None

        resnet = models.resnet34(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_class, kernel_size=4, stride=2, padding=1),
        )
        self.init_criterion()
        self.init_optimizer()

    def init_criterion(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        return True

    def init_optimizer(self):
        if self.opt == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        if self.opt == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.optimizer.zero_grad()
        return True

    def get_desc(self):
        return ''

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def calc_loss(self, outputs, targets, save_loss=True):
        loss = self.criterion(outputs, targets)
        if save_loss:
            self.loss = loss
        return loss

    def backward(self):
        self.loss.backward()
        return True

    def update_params(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        return True


def prepare_data(batch_size=8):
    root = '../data/VOCdevkit/VOC2007'
    train_data = VOCSegmentationDataset(root=root, image_set='train')
    val_data = VOCSegmentationDataset(root=root, image_set='val')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=VOCSegmentationDataset.collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0,
                              collate_fn=VOCSegmentationDataset.collate_fn)
    return train_data, val_data, train_loader, val_loader


def train_model(model, device, data_loader, epoch, logger=None):
    model.train()
    loss = 0

    ix = 0
    total = len(data_loader)
    for batch, (images, masks) in enumerate(data_loader):
        ix += 1
        images, masks = images.to(device), masks.to(device)
        # 4步
        y = model.forward(images)
        # align_corners	是否对齐角点
        y = nn.functional.interpolate(y, size=masks.shape[1:], mode='bilinear', align_corners=True)
        model.calc_loss(y, masks, save_loss=True)
        model.backward()
        model.update_params()

        loss += model.loss.item()
        if logger:
            logger.progress_bar(ix, total)

    loss /= len(data_loader)
    msg = f"Epoch {epoch + 1}, Train Loss: {loss:.4f}"
    if logger:
        logger.log(msg)
    else:
        print(msg)
    return loss


def run():
    # 所有的参数
    model_name = 'fcn'
    batch_size = 8
    opt = 'Adam'
    lr = 1e-4
    epochs = 20
    num_class = 21
    first_clear_log = True

    name_ = f'{model_name}_{opt}_{lr}_{batch_size}_o1'
    loger = Logger(path='results', filename=name_)
    train_loss_ls = []

    st = datetime.datetime.now()
    loger.log(f'start ==========================\nepoch {epochs}', clear=first_clear_log)
    device = get_device(use_gpu=True, loger=loger)
    model = FCN(num_class=num_class, lr=lr, opt=opt).to(device)
    loger.log(f'model:{model_name}')
    loger.log(f'num_class:{num_class}')
    loger.log(f'lr:{model.lr}')
    loger.log(f'opt:{model.opt}')
    loger.log(model.get_desc())

    train_data, test_data, train_loader, test_loader = prepare_data(batch_size=batch_size)

    # loger.log(f'train sample size:{train_data.images[0]}')

    for epoch in range(epochs):
        train_loss = train_model(model, device, train_loader, epoch, logger=loger)
        train_loss_ls.append(train_loss)

    plt_loss_result(train_loss_ls, path='results', name=name_, logger=loger)

    save_model(model, name=name_, path='results')

    ed = datetime.datetime.now()
    loger.log(f'cost {(ed - st).total_seconds()} s')
    loger.log(f'最终结果: 训练损失: {train_loss_ls[-1]:.4f}')
    loger.log('finished.')
    return True


def visualize_fcn_prediction(
        model,
        pic_name,
        image_tensor,
        alpha=0.5,
        show_original=True,
        overlay=True,
) -> None:
    """
    可视化 FCN 的预测结果（支持21类标准VOC配色）
    model: FCN 模型
    image_tensor: 输入图像张量 (1, C, H, W)
    num_classes: 类别数（默认为21）
    alpha: 掩码透明度（0-1）
    show_original: 是否显示原始图像
    overlay: 是否叠加显示掩码
    """
    name = pic_name.split('.')[0]
    save_path = f'results/fcn_seg_{name}.png'
    figsize = (12, 6)

    # --- 2. 模型预测 ---
    with torch.no_grad():
        logits = model(image_tensor)  # (1, 21, H, W)
        pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()  # (H, W)

    # --- 3. 预处理图像 ---
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())  # 归一化到[0,1]

    # --- 4. VOC标准21类颜色映射 ---
    voc_colormap = np.array([
        [0, 0, 0],  # 0: 背景（黑色）
        [128, 0, 0],  # 1: aeroplane（红色）
        [0, 128, 0],  # 2: bicycle（绿色）
        [128, 128, 0],  # 3: bird（黄绿色）
        [0, 0, 128],  # 4: boat（蓝色）
        [128, 0, 128],  # 5: bottle（紫色）
        [0, 128, 128],  # 6: bus（蓝绿色）
        [128, 128, 128],  # 7: car（灰色）
        [64, 0, 0],  # 8: cat（暗红色）
        [192, 0, 0],  # 9: chair（橙红色）
        [64, 128, 0],  # 10: cow（黄绿色）
        [192, 128, 0],  # 11: dining table（橙色）
        [64, 0, 128],  # 12: dog（紫红色）
        [192, 0, 128],  # 13: horse（粉紫色）
        [64, 128, 128],  # 14: motorbike（蓝绿色）
        [192, 128, 128],  # 15: person（肉色）
        [0, 64, 0],  # 16: potted plant（深绿色）
        [128, 64, 0],  # 17: sheep（棕色）
        [0, 192, 0],  # 18: sofa（亮绿色）
        [128, 192, 0],  # 19: train（黄绿色）
        [0, 64, 128]  # 20: tv/monitor（深蓝色）
    ], dtype=np.uint8)

    # 生成彩色掩码
    pred_mask_color = voc_colormap[pred_mask]  # (H, W, 3)

    # 可视化
    plt.figure(figsize=figsize)

    if show_original:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

    if overlay:
        plt.subplot(1, 2, 2 if show_original else 1)
        plt.imshow(image)
        plt.imshow(pred_mask_color, alpha=alpha)
        plt.title("Segmentation Mask (Overlay)")
    else:
        plt.subplot(1, 2, 2 if show_original else 1)
        plt.imshow(pred_mask_color)
        plt.title("Segmentation Mask")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.1)
    # plt.show()
    return

def test_plt():
    model = FCN(num_class=21)
    model.load_state_dict(torch.load('results/fcn_Adam_0.0001_8_o1.pth', map_location='mps'))
    model.to('mps')
    model.eval()  # 如果是推理阶段，设置为eval模式

    pics = ['000032.jpg', '000363.jpg']
    path = '../data/VOCdevkit/VOC2007/JPEGImages'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 归一化
    ])
    for pic in pics:
        image = Image.open(f'{path}/{pic}').convert('RGB')
        image = image.resize((256, 256), Image.BILINEAR)
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to('mps')
        visualize_fcn_prediction(
            model,
            pic_name=pic,
            image_tensor=image_tensor,
            overlay=True,
            alpha=0.7
        )

if __name__ == '__main__':
    # run()
    test_plt()