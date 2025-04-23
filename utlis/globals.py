# -*- utf-8 -*-
"""
放置一些通用函数
"""
import sys


class Logger(object):
    def __init__(self, filename, path):
        self.filename = filename
        self.path = path
    def log(slef, msg, clear=False):
        # 打印并记录日志
        x = '{}/{}'.format(slef.path, slef.filename)
        if clear:
            with open(x,  'w') as f:
                x.close()
        with open(x, 'a') as f:
            f.write(msg + '\n')
            f.close()
        print(msg)
        return True


def progress_bar(current, total, bar_length=50):
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





