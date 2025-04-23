# -*- utf-8 -*-
import numpy as np
from torchvision import datasets, transforms


def prepare_data():
    # 数据加载（MNIST）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_data = datasets.MNIST(root='../data', train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root='../data', train=False, download=False, transform=transform)
    # torch.Size([60000, 28, 28])
    # torch.Size([10000, 28, 28])

    train_X = np.array([x.reshape(-1) for x in np.array(train_data.data)])
    train_Y_labels = np.array(train_data.targets)
    train_Y = np.zeros((train_Y_labels.shape[0], 10))
    for i in range(len(train_Y_labels)):
        train_Y[i][train_Y_labels[i]] = 1

    test_X = np.array([x.reshape(-1) for x in np.array(test_data.data)])
    test_Y_labels = np.array(test_data.targets)
    test_Y = np.zeros((test_Y_labels.shape[0], 10))
    for i in range(len(test_Y_labels)):
        test_Y[i][test_Y_labels[i] ] = 1
    # In [58]: test_X.shape
    # Out[58]: (10000, 784)
    # In [59]: test_Y_labels.shape
    # Out[59]: (10000,)
    # In [60]: test_Y.shape
    # Out[60]: (10000, 10)
    return train_X, train_Y, train_Y_labels, test_X, test_Y, test_Y_labels

class MyNet(object):
    def __init__(self):
        self.lr = 0.01
        # self.batch = 64
        # self.epochs = 10

        self.layer_1_size = 784
        self.layer_2_size = 128
        self.layer_3_size = 10

        self.w1 = np.random.randn(self.layer_1_size, self.layer_2_size) * 0.1
        self.b1 = np.zeros((1, self.layer_2_size))
        self.w2 = np.random.randn(self.layer_2_size, self.layer_3_size) * 0.1
        self.b2 = np.zeros((1, self.layer_3_size))

        self.z1 = None
        self.a1 = None
        self.z2 = None

        # 记录参数的梯度，用于参数更新
        self.dw2 = None
        self.db2 = None
        self.dw1 = None
        self.db1 = None

        self.y = None
        self.loss = None

        self.X = None
        self.Y = None

    # 激活函数和其导数
    @classmethod
    def relu(cls, x):
        return np.maximum(0, x)

    @classmethod
    def relu_derivative(cls, x):
        return (x > 0).astype(float)

    @classmethod
    def softmax(cls, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @classmethod
    def get_loss(cls, y, Y):
        loss = -np.mean(np.sum(Y * np.log(y + 1e-10), axis=1))
        return loss

    # 前向传播
    def forward(self, X, Y):
        self.X = X
        self.Y = Y
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.y = self.softmax(self.z2)
        self.loss = self.get_loss(self.y, self.Y)
        return

    def backward(self):
        # 反向传播
        batch_size = self.X.shape[0]
        dz2 = self.y - self.Y
        self.dw2 = np.dot(self.a1.T, dz2) / batch_size
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / batch_size

        dz1 = np.dot(dz2, self.w2.T) * self.relu_derivative(self.z1)
        self.dw1 = np.dot(self.X.T, dz1) / batch_size
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / batch_size
        return

    def optimize_params(self):
        # 更新参数
        self.w1 -= self.lr * self.dw1
        self.b1 -= self.lr * self.db1
        self.w2 -= self.lr * self.dw2
        self.b2 -= self.lr * self.db2
        return


def train_mod(train_X, train_Y, train_Y_labels):
    # train_X, train_Y, train_Y_labels, test_X, test_Y, test_Y_labels =  prepare_data()
    epochs = 80
    batch = 100
    # 初始化网络
    mod = MyNet()
    for epoch in range(epochs):
        # 打乱数据
        indices = np.arange(train_X.shape[0])
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_Y = train_Y[indices]

        for i in range(0, train_X.shape[0], batch):
            # 获取小批量数据
            x_batch = train_X[i:i + batch]
            y_batch = train_Y[i:i + batch]
            mod.forward(x_batch, y_batch)
            mod.backward()
            mod.optimize_params()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {mod.loss:.4f}')
    return mod

def test_mod(mod, test_X, test_Y, test_Y_labels):
    # 测试模型
    # X = test_X[0:3]
    # Y = test_Y[0:3]
    mod.forward(test_X, test_Y)

    # 计算准确率
    predictions = np.argmax(mod.y, axis=1)
    labels = np.argmax(test_Y, axis=1)
    accuracy = np.mean(predictions == labels)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    train_X, train_Y, train_Y_labels, test_X, test_Y, test_Y_labels = prepare_data()
    mod = train_mod(train_X, train_Y, train_Y_labels)
    test_mod(mod, test_X, test_Y, test_Y_labels)
