from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from deal_data import SmallPicData
from utlis.globals import save_model, load_model
from models.lenet import MyLeNet

def get_small_img(model):
    data_path = '../data/a-i-2025-01/yanzhengma_train/sub_train'
    my_data = SmallPicData(
        data_path, batch_size=1, test_ratio=0.2, random_state=6,
        is_subset=False, subset_percent=0.1, logger=None
    )
    img = my_data.test_dataset[0][0]
    # 加上batch
    img = img.unsqueeze(0)
    return img

def plot_feature_map(feats, title, path='results', name='test', n=8):
    """
    feats: list of tensors, 每个形状为 (C, H, W)
    title: 图像标题
    path: 保存路径
    name: 文件名前缀
    n: 每个特征绘制的通道数（前n个通道）
    """
    os.makedirs(path, exist_ok=True)

    num_feats = len(feats)
    # 确保n不超过任意特征的通道数
    n = min(n, min(feat.shape[0] for feat in feats))

    plt.figure(figsize=(n * 2, num_feats * 2))  # 宽度根据通道数，行高根据特征数

    for f_idx, feat in enumerate(feats):
        for c_idx in range(n):
            plt_idx = f_idx * n + c_idx + 1
            plt.subplot(num_feats, n, plt_idx)
            channel_img = feat[c_idx].cpu().numpy()
            plt.imshow(channel_img, cmap='viridis')
            plt.axis('on')

            # 第一行显示通道号标题
            if f_idx == 0:
                plt.title(f'Ch {c_idx}', fontsize=8)
            # 第一列显示特征编号
            if c_idx == 0:
                plt.ylabel(f'Feat {f_idx}', fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 给标题留点空间
    save_path = os.path.join(path, f'{name}_feat_{title}.png')
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"特征图已保存到 {save_path}")


if __name__ == '__main__':
    name_ = 'lenet_Adam_0.001_64_1.pth'
    model_class = MyLeNet
    model = load_model(MyLeNet, path=f'results/{name_}')
    print(f'{name_} ')

    features = {}
    def get_activation(name):
        def hook(model, input, output):
            features[name] = output.detach().cpu()
        return hook

    f_funcs = model.get_plot_feature_hooks()
    f_names = []
    handles = []
    for index in range(len(f_funcs)):
        # 把想要的特征图注册进来
        x = f'{model.m_name}_{index}'
        f_names.append(x)
        handles.append(f_funcs[index].register_forward_hook(get_activation(x)))

    # run 图
    device = torch.device("mps")
    img = get_small_img(model).to(device)
    with torch.no_grad():
        output = model(img)

    # 取出特征图
    feats = []
    for ix in range(len(f_names)):
        feats.append(features[f_names[ix]][0])

    plot_feature_map(feats, title=name_, path='results', name=name_.replace('.pth', ''), n=8)

