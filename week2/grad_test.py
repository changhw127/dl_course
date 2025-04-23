# -*- coding: utf-8 -*-
import copy
import numpy as np


class Net(object):
    def __init__(self, X, Y):
        # 初始化参数
        self.X = X
        self.Y = Y
        self.w1 = np.array([[1, 1], [1, -1]])
        self.b1 = np.ones((2, 1))
        self.w2 = np.array([1, -1])
        self.b2 = np.ones(1)
        self.lr = 0.01

        # 前向传播记录
        self.z = None
        self.a = None
        self.a_grad = None
        self.y = None
        self.loss = None

    # 定义relu
    def relu(self, z):
        z[z < 0] = 0
        return z

    def relu_grad(self, a):
        # 这一批样本的a1，a2，如果a1中有>0。则梯度为1，全是0，则梯度为0
        a_grad = copy.deepcopy(a)
        a_grad = np.sum(a_grad, axis=1, keepdims=False)
        a_grad[a_grad > 0] = 1
        a_grad[a_grad <= 0] = 0
        return a_grad

    # 前向传播
    def froward(self):
        self.z = self.w1 @ self.X.T + self.b1
        self.a = self.relu(self.z)
        self.y = self.w2 @ self.a + self.b2
        self.loss = pow(self.y - self.Y, 2) / 2
        self.a_grad = self.relu_grad(self.a)
        return self.y, self.loss


if __name__ == '__main__':
    # 初始化 X , Y
    # X = np.random.randn(10, 2)
    # Y = np.random.randn(10)
    # 方便重复实验
    X = np.array([[0.06992709, 0.11076548],
                  [0.10591315, 1.80070395],
                  [0.55563535, 2.38782628],
                  [0.02004028, -0.29057906],
                  [-0.57128551, 0.86811334],
                  [-1.37511582, -0.01367848],
                  [-0.87340918, -1.52586694],
                  [0.50031181, -0.70032277],
                  [-0.18833502, -0.54190003],
                  [0.490605, 0.3643706]])
    Y = np.array([-1.70054165, 0.09400653, 0.44345448, 0.63314569, 2.38174948,
                  -0.47089433, 1.1641137, 1.27154141, 1.50590629, 3.25775702])
    # 初始化网络
    net = Net(X, Y)
    # 前向传播 + 求损失 + 求da/dz的梯度
    net.froward()
    w11_grad = (net.y - Y) * net.w2[0] * net.a_grad[0] @ X[:, 0]
    # 1.429320300299429
    w21_grad = (net.y - Y) * net.w2[1] * net.a_grad[1] @ X[:, 0]
    # -1.429320300299429

