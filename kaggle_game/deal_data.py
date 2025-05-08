# -*- utf-8 -*-
import torch
import os
import cv2
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from utlis.globals import *


"""
加载数据，直接加载，或者加载拆分开的数据
"""
def denoise_captcha_img(img):
    """
    输入: img (PIL.Image 或 numpy.ndarray, 彩色)
    输出: denoised_img (numpy.ndarray, shape: HxW, dtype: uint8)
    """
    # 转为灰度
    if isinstance(img, Image.Image):
        img = np.array(img)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # PIL默认是RGB
    else:
        gray = img.copy()

    # 判断背景，自动反转
    mean_val = np.mean(gray)
    if mean_val > 127:
        gray = 255 - gray

    # 自适应二值化
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 去点状噪声
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 去线状噪声
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    denoised_img = cv2.morphologyEx(closing, cv2.MORPH_OPEN, line_kernel, iterations=1)

    return denoised_img

def sobel_edge(img):
    # 1. PIL.Image转numpy数组
    img_np = np.array(img)  # shape: (H, W, 3), dtype: uint8

    # 2. 转为灰度
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    ray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Sobel算子
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # 4. 归一化到0~255并转uint8
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    sobel = sobel.astype(np.uint8)

    # 5. 转回PIL.Image
    return Image.fromarray(sobel)

def auto_canny(img, sigma=0.33):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    v = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    lower = 50
    upper = 200
    edges = cv2.Canny(gray, lower, upper)
    return Image.fromarray(edges)

def canny_edge(img, low_threshold=50, high_threshold=150):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return Image.fromarray(edges)

def get_gaussian_gray(img, ksize=3, sigma=0):
    # 1. PIL.Image转numpy数组
    img_np = np.array(img)
    # 2. 转为灰度
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # 3. 高斯模糊
    gray_blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    # 4. 转回PIL.Image
    return Image.fromarray(gray_blur)

def denoise_captcha_color_img(img):
    """
    输入: img (PIL.Image 或 numpy.ndarray, 彩色)
    输出: denoised_img (numpy.ndarray, 彩色, shape: HxWx3)
    """
    # 转为numpy数组
    if isinstance(img, Image.Image):
        img = np.array(img)
    # PIL默认是RGB，cv2默认是BGR，注意区分
    # 这里我们保持RGB

    # 1. 中值滤波去除椒盐噪声
    denoised = cv2.medianBlur(img, 3)

    # 2. 可选：形态学操作去除小噪点
    kernel = np.ones((2,2), np.uint8)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel, iterations=1)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=1)

    return Image.fromarray(denoised)


class SinglePicDataset(Dataset):
    def __init__(self, root_dir, transform, use_denoise=True, use_sobel=True):
        """
        root_dir: 图片文件夹路径
        transform: 对单个30x30小图的transform
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_denoise = use_denoise
        self.use_sobel = use_sobel
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # 收集所有标签字符，建立映射
        # all_labels = []
        # for f in self.image_files:
        #     base_name = os.path.splitext(f)[0]
        #     all_labels.extend(list(base_name))
        # self.classes = sorted(list(set(all_labels)))
        # self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.classes = self.get_classes()
        self.class_to_idx = self.get_class_to_index()

    def __len__(self):
        # 总样本数是图片数 * 5（每张图拆成5个小图）
        return len(self.image_files) * 5

    def __getitem__(self, idx):
        # idx对应第几张图片的第几个小图
        # 先确定是哪张图片
        img_idx = idx // 5
        sub_idx = idx % 5  # 0~4，表示第几个30x30小图

        img_name = self.image_files[img_idx]
        img_path = os.path.join(self.root_dir, img_name)

        # 读取整张30x150图片
        img1 = Image.open(img_path).convert('RGB')

        # 拆分成5个30x30小图，水平拼接
        left = sub_idx * 30
        right = left + 30
        sub_img1 = img1.crop((left, 0, right, 30))  # (left, upper, right, lower)

        base_name = os.path.splitext(img_name)[0]
        label_char = base_name[sub_idx]

        if self.use_denoise:
            img2 = Image.open(img_path).convert('RGB')
            img2 = denoise_captcha_color_img(img2)
            left = sub_idx * 30
            right = left + 30
            sub_img2 = img2.crop((left, 0, right, 30))  # (left, upper, right, lower)

        if self.use_sobel:
            img3 = Image.open(img_path).convert('RGB')
            img3 = sobel_edge(img3)
            left = sub_idx * 30
            right = left + 30
            sub_img3 = img3.crop((left, 0, right, 30))  # (left, upper, right, lower)

        if self.transform:
            sub_img = self.transform(sub_img1)

        label_idx = self.class_to_idx[label_char]
        return sub_img, label_idx

    @classmethod
    def get_classes(cls):
        return ['0',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                'A',
                'B',
                'C',
                'D',
                'E',
                'F',
                'G',
                'H',
                'I',
                'J',
                'K',
                'L',
                'M',
                'N',
                'O',
                'P',
                'Q',
                'R',
                'S',
                'T',
                'U',
                'V',
                'W',
                'X',
                'Y',
                'Z',
                'a',
                'b',
                'c',
                'd',
                'e',
                'f',
                'g',
                'h',
                'i',
                'j',
                'k',
                'l',
                'm',
                'n',
                'o',
                'p',
                'q',
                'r',
                's',
                't',
                'u',
                'v',
                'w',
                'x',
                'y',
                'z']

    @classmethod
    def get_class_to_index(cls):
        return {'0': 0,
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6,
                '7': 7,
                '8': 8,
                '9': 9,
                'A': 10,
                'B': 11,
                'C': 12,
                'D': 13,
                'E': 14,
                'F': 15,
                'G': 16,
                'H': 17,
                'I': 18,
                'J': 19,
                'K': 20,
                'L': 21,
                'M': 22,
                'N': 23,
                'O': 24,
                'P': 25,
                'Q': 26,
                'R': 27,
                'S': 28,
                'T': 29,
                'U': 30,
                'V': 31,
                'W': 32,
                'X': 33,
                'Y': 34,
                'Z': 35,
                'a': 36,
                'b': 37,
                'c': 38,
                'd': 39,
                'e': 40,
                'f': 41,
                'g': 42,
                'h': 43,
                'i': 44,
                'j': 45,
                'k': 46,
                'l': 47,
                'm': 48,
                'n': 49,
                'o': 50,
                'p': 51,
                'q': 52,
                'r': 53,
                's': 54,
                't': 55,
                'u': 56,
                'v': 57,
                'w': 58,
                'x': 59,
                'y': 60,
                'z': 61}


class SinglePicDatasetOld(Dataset):
    def __init__(self, root_dir, transform, use_denoise=True):
        """
        root_dir: 图片文件夹路径
        transform: 对单个30x30小图的transform
        """
        self.root_dir = root_dir
        self.transform = transform
        self.use_denoise = use_denoise
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        # 收集所有标签字符，建立映射
        # all_labels = []
        # for f in self.image_files:
        #     base_name = os.path.splitext(f)[0]
        #     all_labels.extend(list(base_name))
        # self.classes = sorted(list(set(all_labels)))
        # self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.classes = self.get_classes()
        self.class_to_idx = self.get_class_to_index()

    def __len__(self):
        # 总样本数是图片数 * 5（每张图拆成5个小图）
        return len(self.image_files) * 5

    def __getitem__(self, idx):
        # idx对应第几张图片的第几个小图
        # 先确定是哪张图片
        img_idx = idx // 5
        sub_idx = idx % 5  # 0~4，表示第几个30x30小图

        img_name = self.image_files[img_idx]
        img_path = os.path.join(self.root_dir, img_name)

        # 读取整张30x150图片
        img = Image.open(img_path).convert('RGB')
        if self.use_denoise:
            img = denoise_captcha_color_img(img)

        # 拆分成5个30x30小图，水平拼接
        left = sub_idx * 30
        right = left + 30
        sub_img = img.crop((left, 0, right, 30))  # (left, upper, right, lower)

        base_name = os.path.splitext(img_name)[0]
        label_char = base_name[sub_idx]

        if self.transform:
            sub_img = self.transform(sub_img)

        label_idx = self.class_to_idx[label_char]
        return sub_img, label_idx

    @classmethod
    def get_classes(cls):
        return ['0',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                'A',
                'B',
                'C',
                'D',
                'E',
                'F',
                'G',
                'H',
                'I',
                'J',
                'K',
                'L',
                'M',
                'N',
                'O',
                'P',
                'Q',
                'R',
                'S',
                'T',
                'U',
                'V',
                'W',
                'X',
                'Y',
                'Z',
                'a',
                'b',
                'c',
                'd',
                'e',
                'f',
                'g',
                'h',
                'i',
                'j',
                'k',
                'l',
                'm',
                'n',
                'o',
                'p',
                'q',
                'r',
                's',
                't',
                'u',
                'v',
                'w',
                'x',
                'y',
                'z']

    @classmethod
    def get_class_to_index(cls):
        return {'0': 0,
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6,
                '7': 7,
                '8': 8,
                '9': 9,
                'A': 10,
                'B': 11,
                'C': 12,
                'D': 13,
                'E': 14,
                'F': 15,
                'G': 16,
                'H': 17,
                'I': 18,
                'J': 19,
                'K': 20,
                'L': 21,
                'M': 22,
                'N': 23,
                'O': 24,
                'P': 25,
                'Q': 26,
                'R': 27,
                'S': 28,
                'T': 29,
                'U': 30,
                'V': 31,
                'W': 32,
                'X': 33,
                'Y': 34,
                'Z': 35,
                'a': 36,
                'b': 37,
                'c': 38,
                'd': 39,
                'e': 40,
                'f': 41,
                'g': 42,
                'h': 43,
                'i': 44,
                'j': 45,
                'k': 46,
                'l': 47,
                'm': 48,
                'n': 49,
                'o': 50,
                'p': 51,
                'q': 52,
                'r': 53,
                's': 54,
                't': 55,
                'u': 56,
                'v': 57,
                'w': 58,
                'x': 59,
                'y': 60,
                'z': 61}


class SmallPicData(object):
    def __init__(self, path, batch_size, test_ratio=0.2, random_state=2,
                 is_subset=False, subset_percent=0.2,
                 logger=None):
        self.path = path
        self.batch_size = batch_size
        self.test_ratio = test_ratio
        self.full_dataset = None
        self.random_state = random_state
        self.is_subset = is_subset
        self.subset_percent = subset_percent
        self.logger = logger
        self._load_data()

    def img_show(self, dataset, index):
        # 查看数据
        # dataset[0]
        data = dataset[index]
        image, label_index = data
        plt.imshow(image.permute(1, 2, 0))  # 从(C,H,W)转为(H,W,C)
        plt.axis('off')
        plt.show()
        print(f"标签: {self.full_dataset.classes[label_index]}")
        return True

    def _transform(self):
        # 定义图像预处理流程
        transform = transforms.Compose([
            # 1. 先调整大小：将图像的短边缩放，长边按比例缩放
            # transforms.Resize(self.resize + 2),
            # # 2. 然后中心裁剪：确保所有图像都是指定size
            # transforms.CenterCrop(self.resize),
            # 3. 转换为Tensor并自动归一化到[0,1]
            transforms.ToTensor(),
            # 可选：标准化（使用ImageNet的均值和标准差）
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform

    def _load_data(self):
        self.full_dataset = SinglePicDataset(self.path, self._transform())
        labels = [label for _, label in self.full_dataset]

        # 使用分层抽样划分
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_ratio, random_state=self.random_state)
        train_indices, test_indices = next(sss.split(np.zeros(len(labels)), labels))
        self.train_dataset = torch.utils.data.Subset(self.full_dataset, train_indices)
        self.test_dataset = torch.utils.data.Subset(self.full_dataset, test_indices)

        if self.is_subset:
            # 默认获取 1% 的数据用于拉通
            indices = torch.randperm(len(self.train_dataset))[:int(len(self.train_dataset) * self.subset_percent)]
            self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)
            indices = torch.randperm(len(self.test_dataset))[:int(len(self.test_dataset) * self.subset_percent)]
            self.test_dataset = torch.utils.data.Subset(self.test_dataset, indices)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 训练集需要打乱
            pin_memory=True  # 加速GPU传输
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1000,
            shuffle=False,  # 测试集不需要打乱
        )
        msg = f"train_data: {len(self.train_loader.dataset)}, test_data: {len(self.test_loader.dataset)}, batch_size: {self.batch_size}"
        if self.logger:
            self.logger.log(msg)
        else:
            print(msg)
        return True




def show_sub_images(img_path):
    img = Image.open(img_path).convert('RGB')
    sub_images = []
    for i in range(5):
        left = i * 30
        upper = 0
        right = left + 30
        lower = 30
        sub_img = img.crop((left, upper, right, lower))
        sub_images.append(sub_img)

    # 画图
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(sub_images[i])
        ax.set_title(f'Sub-image {i+1}')
        ax.axis('off')
    plt.show()

def plt_denosie_img():
    root_dir = '../data/a-i-2025-01/yanzhengma_train/sub_train'
    name = '0AyAv.jpg'
    img_path = f'{root_dir}/{name}'
    img = Image.open(img_path).convert('RGB')
    # 处理
    denoised_img = denoise_captcha_color_img(img)

    # 画图对比
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Denoised')
    plt.imshow(denoised_img, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plt_denoise_imgs(img_names, root_dir, denoise_func):
    n = len(img_names)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2 * n))
    if n == 1:
        axes = [axes]  # 保证可迭代

    for i, name in enumerate(img_names):
        img_path = os.path.join(root_dir, name)
        img = Image.open(img_path).convert('RGB')
        denoised_img = denoise_func(img)

        # 原图
        axes[i][0].imshow(img)
        axes[i][0].set_title('Original')
        axes[i][0].axis('off')

        # 去噪图
        # 如果denoised_img是numpy数组，且shape为(H,W,3)，可以直接imshow
        axes[i][1].imshow(denoised_img)
        axes[i][1].set_title('Denoised')
        axes[i][1].axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig('analysis/denosie_result.png', dpi=100)

def plt_solbel_imgs(img_names, root_dir, func):
    n = len(img_names)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2 * n))
    if n == 1:
        axes = [axes]  # 保证可迭代

    for i, name in enumerate(img_names):
        img_path = os.path.join(root_dir, name)
        img = Image.open(img_path).convert('RGB')
        sobel_img = func(img)

        # 原图
        axes[i][0].imshow(img)
        axes[i][0].set_title('Original')
        axes[i][0].axis('off')

        # 去噪图
        # 如果denoised_img是numpy数组，且shape为(H,W,3)，可以直接imshow
        axes[i][1].imshow(sobel_img)
        axes[i][1].set_title('sobel')
        axes[i][1].axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig('analysis/sobel_result.png', dpi=100)

def plt_deal_imgs(img_names, root_dir):
    n = len(img_names)
    fig, axes = plt.subplots(n, 5, figsize=(10, 2 * n))
    if n == 1:
        axes = [axes]  # 保证可迭代

    for i, name in enumerate(img_names):
        img_path = os.path.join(root_dir, name)
        img1 = Image.open(img_path).convert('RGB')
        img2 = denoise_captcha_color_img(img1)
        img3 = sobel_edge(img2)
        img4 = canny_edge(img2)
        img5 = auto_canny(img1)

        # 原图
        axes[i][0].imshow(img1)
        axes[i][0].set_title('Original')
        axes[i][0].axis('off')

        # 去噪图
        axes[i][1].imshow(img2)
        axes[i][1].set_title('denoise')
        axes[i][1].axis('off')

        # 去噪图的边缘图
        axes[i][2].imshow(img3)
        axes[i][2].set_title('sobel')
        axes[i][2].axis('off')

        axes[i][3].imshow(img4)
        axes[i][3].set_title('canny')
        axes[i][3].axis('off')

        axes[i][4].imshow(img2)
        axes[i][4].set_title('gauss')
        axes[i][4].axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig('analysis/deal_result.png', dpi=100)



def test1():
    path = '../data/a-i-2025-01/yanzhengma_train/sub_train'
    my_data = SmallPicData(path, batch_size=64, test_ratio=0.2, random_state=6, is_subset=False, subset_percent=0.2)
    train_data, test_data, train_loader, test_loader = my_data.train_dataset, my_data.test_dataset, my_data.train_loader, my_data.test_loader
    # print(f"train_data: {len(train_loader.dataset)}, test_data: {len(test_loader.dataset)}, batch_size: {my_data.batch_size}")

    # 求一下数据的标签数

    idx = 111
    my_data.img_show(train_data, idx)
    print(train_data[idx][0].shape)
    return

def test2():
    root_dir = '../data/a-i-2025-01/yanzhengma_train/sub_train'
    files  = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    files_select = random.sample(files, 10)
    plt_denoise_imgs(files_select, root_dir, denoise_captcha_color_img)

def test3():
    root_dir = '../data/a-i-2025-01/yanzhengma_train/sub_train'
    files  = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
    img_names = random.sample(files, 10)
    plt_solbel_imgs(img_names, root_dir, sobel_edge)


def test():
    root_dir = '../data/a-i-2025-01/yanzhengma_train/sub_train'
    dataset = SinglePicDataset(root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
    for imgs, labels in dataloader:
        print(imgs.shape)  # [batch_size, 3, 30, 30]
        print(labels)      # 对应的标签列表
        break

    name = '0a5Jz.jpg'
    img_name = f'{root_dir}/{name}'
    show_sub_images(img_name)
    return