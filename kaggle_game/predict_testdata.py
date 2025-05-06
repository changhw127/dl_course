#  -*- utf-8 -*-
"""
1 根据模型生成target_数据
2 尝试在所有训练集上进行演算
"""
import os
import pandas as pd
import torch
from PIL import Image
from deal_data import SmallPicData, SinglePicDataset, denoise_captcha_color_img
from utlis.globals import load_model, Logger
from models.lenet import MyLeNet
from torchvision import datasets, transforms
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor(),
    # 可选：标准化（使用ImageNet的均值和标准差）
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_small_pic(path, file, use_denoise=True):
    # 返回tensors 以及文件名
    img_path = os.path.join(path, file)
    img = Image.open(img_path).convert('RGB')
    if use_denoise:
        img = denoise_captcha_color_img(img)
    imgs = []
    for sub_idx in range(5):
        left = sub_idx * 30
        right = left + 30
        sub_img = img.crop((left, 0, right, 30))
        sub_img = transform(sub_img)
        imgs.append(sub_img)
    imgs = torch.stack(imgs)
    return imgs, file.split('.')[0]


def generate_test_result(name_, model_class):
    # name_ = 'lenet_Adam_0.001_64.pth'
    path = '../data/a-i-2025-01/yanzhengma_test/test'
    # model_class = MyLeNet
    model = load_model(model_class, path=f'results/{name_}')

    logger = Logger(path='pred', filename='generate_test_result')
    image_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    classes = SinglePicDataset.get_classes()
    device = torch.device("mps")
    records = []
    ix = 0
    for file in image_files:
        ix += 1
        if ix %100 == 0:
            print(ix)
        imgs, label = get_small_pic(path, file)
        # imgs.shape torch.Size([5, 3, 30, 30])
        imgs =imgs.to(device)
        with torch.no_grad():
            output = model(imgs)
            probs = F.softmax(output, dim=0)
            pred_label_indexes = [torch.argmax(p).item() for p in probs]
            p_label = ''.join([classes[idx] for idx in pred_label_indexes])
            records.append({
                # 'file': file,
                'id': label,
                'y': p_label,
            })
    df = pd.DataFrame.from_records(records)
    total = len(df)
    df.to_csv(f'pred/{name_}.pred_test.csv', index=None)
    msg = f'model:{name_}\ttotal:{total}'
    logger.log(f'[{logger.dt_str()}] {msg}')
    return


def generate_train_result(name_, model_class):
    # name_ = 'lenet_Adam_0.001_64.pth'
    path = '../data/a-i-2025-01/yanzhengma_train/train'
    # model_class = MyLeNet
    model = load_model(model_class, path=f'results/{name_}')

    logger = Logger(path='pred', filename='generate_train_result')
    image_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    classes = SinglePicDataset.get_classes()
    device = torch.device("mps")
    records = []
    ix = 0
    for file in image_files:
        ix += 1
        if ix %100 == 0:
            print(ix)
        imgs, label = get_small_pic(path, file)
        # imgs.shape torch.Size([5, 3, 30, 30])
        imgs =imgs.to(device)
        with torch.no_grad():
            output = model(imgs)
            probs = F.softmax(output, dim=0)
            pred_label_indexes = [torch.argmax(p).item() for p in probs]
            p_label = ''.join([classes[idx] for idx in pred_label_indexes])
            records.append({
                'file': file,
                'real_target': label,
                'pred_target': p_label,
                'is_same': 1 if label == p_label else 0,
            })
    df = pd.DataFrame.from_records(records)
    total = len(df)
    same_n = sum(df['is_same'])
    acc = same_n * 100.0 / total
    df.to_csv(f'pred/{name_}.pred_train.csv', index=None)
    msg = f'model:{name_}\tacc:{acc} % \tsame:{same_n}\ttotal:{total}'
    logger.log(f'[{logger.dt_str()}] {msg}')
    return True

if __name__ == '__main__':
    from models.lenet import MyLeNet1
    from models.vgg import MyVGG1
    name = 'vgg_Adam_0.001_64.pth'

    generate_train_result(name, MyVGG1)
    generate_test_result(name, MyVGG1)