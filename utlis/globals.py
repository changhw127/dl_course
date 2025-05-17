# -*- utf-8 -*-
"""
放置一些通用函数
"""
import sys
import copy
import torch
import matplotlib.pyplot as plt
import os
import shutil
import random


class Logger(object):
    def __init__(self, path, filename):
        self.filename = '{}.log'.format(filename)
        self.path = path
    def log(self, msg, clear=False, show=True):
        # 打印并记录日志
        x = '{}/{}'.format(self.path, self.filename)
        if clear:
            with open(x,  'w') as f:
                f.close()
        with open(x, 'a') as f:
            f.write(msg + '\n')
            f.close()
        if show:
            print(msg)
        return True

    @classmethod
    def progress_bar(cls, current, total, bar_length=50):
        # 用来打印进度条
        percent = float(current) / total
        arrow = '=' * int(round(percent * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write(f"\r进度: [{arrow + spaces}] {int(round(percent * 100))} %")
        sys.stdout.flush()

        if current == total:
            sys.stdout.write('\n')
            sys.stdout.flush()
        return True


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): 容忍多少个epoch验证指标不提升后停止训练
            verbose (bool): 是否打印早停信息
            delta (float): 验证指标提升的最小阈值
            path (str): 保存最佳模型参数的路径
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss  # 这里以验证损失为例，损失越小越好

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping计数器: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''验证指标提升时保存模型参数'''
        if self.verbose:
            print(f"验证损失下降 ({self.val_loss_min:.6f} --> {val_loss:.6f})，保存模型参数...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_device(use_gpu=True, loger=None):
    # 选择是否使用GPU加速
    if torch.backends.mps.is_available() and use_gpu:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if loger:
        loger.log(f"Using device: {device}")
    return device


def plt_result(data, path='', name='test', logger=None):
    train_loss_ls, train_acc_ls, test_loss_ls, test_acc_ls = data
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    train_loss = [t.cpu().item() if torch.is_tensor(t) else t for t in train_loss_ls]
    test_loss = [t.cpu().item() if torch.is_tensor(t) else t for t in test_loss_ls]

    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_ls, label='Train Accuracy')
    plt.plot(test_acc_ls, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    # plt.show()
    plt.savefig('{}/{}.png'.format(path, name), dpi=300)
    if logger:
        logger.log('result plt ok.')
    return True


def plt_loss_result(data, path='', name='test', logger=None):
    plt.figure(figsize=(6, 4))
    train_loss = [t.cpu().item() if torch.is_tensor(t) else t for t in data]

    plt.plot(train_loss, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('{}/{}.png'.format(path, name), dpi=300)
    if logger:
        logger.log('result plt ok.')
    return True



def load_model(model_class, path, device='mps'):
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()  # 如果是推理阶段，设置为eval模式
    print(f"模型参数已从 {path} 加载")
    return model


def save_model(model, name, path='results'):
    torch.save(model.state_dict(), '{}/{}.pth'.format(path, name))
    print(f"模型参数已保存到 {path}")


def copy_random_files(src_dir, dst_dir, percent=None, nums=None):
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 获取源目录下所有文件（不包括子目录）
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # 计算10%的文件数量，至少复制1个文件（如果文件数大于0）
    num_to_copy = 0
    if percent:
        num_to_copy = max(1, int(len(files) * percent)) if files else 0
    if nums:
        num_to_copy = nums

    # 随机选择文件
    files_to_copy = random.sample(files, num_to_copy)

    # 复制文件
    for filename in files_to_copy:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copy2(src_path, dst_path)
        print(f"Copied {filename} to {dst_dir}")
    return True


