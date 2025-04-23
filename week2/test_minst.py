import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torchvision import datasets, transforms


def prepare_data():
    # 定义数据转换（归一化）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为 Tensor，并归一化到 [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # 标准化到 [-1, 1]
    ])

    # 加载训练集和测试集
    train_data = datasets.MNIST(
        root='./../data',  # 数据存储路径
        train=True,     # 加载训练集
        download=True,  # 如果本地没有，则下载
        transform=transform
    )

    test_data = datasets.MNIST(
        root='./../data',
        train=False,    # 加载测试集
        download=True,
        transform=transform
    )

    print("训练集样本数:", len(train_data))
    print("测试集样本数:", len(test_data))
    print("图片形状:", train_data[0][0].shape)  # torch.Size([1, 28, 28])
    # print("类别标签:", train_data[0][1])  # 数字 0~9

    return train_data, test_data

# 1. 加载MNIST数据集
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# 数据预处理：归一化 + One-Hot编码
X = X / 255.0  # 归一化到[0,1]
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 2. 定义神经网络结构
input_size = 784  # 28x28
hidden_size = 128  # 隐藏层神经元数量
output_size = 10  # 10类数字
learning_rate = 0.1
batch_size = 64
epochs = 20

# 初始化参数（He初始化）
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
b2 = np.zeros(output_size)


# 3. 定义激活函数和损失函数
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / m  # 加1e-8防止log(0)


# 4. 小批量梯度下降训练
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, X_train.shape[0], batch_size):
        # 获取小批量数据
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # 前向传播
        z1 = np.dot(X_batch, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = softmax(z2)

        # 计算损失
        loss = cross_entropy_loss(y_pred, y_batch)
        epoch_loss += loss

        # 反向传播
        dL_dz2 = y_pred - y_batch  # 输出层梯度
        dL_dW2 = np.dot(a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0)

        dL_da1 = np.dot(dL_dz2, W2.T)
        dL_dz1 = dL_da1 * relu_derivative(z1)
        dL_dW1 = np.dot(X_batch.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0)

        # 参数更新
        W1 -= learning_rate * dL_dW1 / batch_size
        b1 -= learning_rate * dL_db1 / batch_size
        W2 -= learning_rate * dL_dW2 / batch_size
        b2 -= learning_rate * dL_db2 / batch_size

    # 每个epoch打印损失
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / (X_train.shape[0] // batch_size):.4f}")


# 5. 模型评估
def predict(X):
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    return np.argmax(softmax(z2), axis=1)


# 测试集准确率
y_pred = predict(X_test)
accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")