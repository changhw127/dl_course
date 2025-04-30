from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from deal_data import MyData



def get_activation(name):
    def hook(model, input, output):
        features[name] = output.detach().cpu()
    return hook

def load_model(name, path='try/results1'):
    model = torch.load('{}/{}.pth'.format(path, name))
    model.eval()
    return model

# def get_img():
#     x = Image.open('../data/course/faces/female/' + '111.jpg').convert('RGB')
#     x = np.array(x)  # (H, W, C)
#     x = x.transpose((2, 0, 1))  # (C, H, W)
#     img = torch.tensor(x, dtype=torch.float32) / 255.0  # 归一化到[0,1]
#     img = img.unsqueeze(0)  # (1, 3, 100, 100)
#     # 如果训练时有mean/std归一化，这里加上
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
#     img = (img - mean) / std
#     return img

def get_img(model):
    my_data = MyData('../data/course/faces', batch_size=64, resize=model.raw_size,
                     test_ratio=0.2, random_state=6,
                     is_subset=False, subset_percent=0.1, logger=None)
    img = my_data.test_dataset[0][0]
    # 加上batch
    img = img.unsqueeze(0)
    return img


def plot_feature_map(feat, title, path='try/results1', name=''):
    plt.figure(figsize=(6, 3))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(feat[i].numpy(), cmap='viridis')
        plt.axis('off')
    plt.suptitle(title)
    # plt.show()
    plt.savefig(f'{path}/{name}_feat_{title}.png', dpi=100)

def get_features(name):
    # 'lenet', 'alexnet', 'vgg', 'nin', 'googlenet', 'resnet'
    name_ = name.split('_')[0]
    if name_ == 'lenet':
        return model.model[0], model.model[5]
    if name_ == 'alexnet':
        return model.model[0], model.model[12]
    if name_ == 'vgg':
        return model.features[0], model.features[7]
    if name_ == 'nin':
        return model.features[0], model.features[6]
    if name_ == 'googlenet':
        return model.features[0], model.features[14]
    if name_ == 'resnet':
        return model.conv1, model.layer4

if __name__ == '__main__':
    names = [
        'lenet_SGD_0.01_64',
        'alexnet_SGD_0.01_64',
        'vgg_SGD_0.01_64',
        'nin_SGD_0.01_64',
        'googlenet_SGD_0.01_64',
        'resnet_SGD_0.01_64',
    ]
    for name_ in names:
        model = load_model(name_, path='results')
        print(f'{name_} ')
        features = {}
        f1, f2 = get_features(name_)
        handle1 = f1.register_forward_hook(get_activation('shallow'))  # 浅层
        handle2 = f2.register_forward_hook(get_activation('deep'))

        device = torch.device("mps")
        img = get_img(model).to(device)
        with torch.no_grad():
            output = model(img)

        # 取出特征图
        feat_first = features['shallow'][0]
        feat_last = features['deep'][0]

        # 可视化前n个通道
        plot_feature_map(feat_first, 'shallow', path='results', name=name_)
        plot_feature_map(feat_last, 'deep', path='results', name=name_)

