# -*- utf-8 -*-
"""
放置一些通用函数
"""
import sys
import torch
import matplotlib.pyplot as plt


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


