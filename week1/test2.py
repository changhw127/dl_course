import numpy as np
import matplotlib.pyplot as plt

# 定义损失函数
def loss(y, y_hat):
    return - (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

# 生成预测值范围 (0.01 到 0.99，避免 log(0))
y_hat = np.linspace(0.01, 0.99, 100)

# 计算 y=1 和 y=0 时的损失
loss_y1 = loss(1, y_hat)
loss_y0 = loss(0, y_hat)

# 绘制图像
plt.figure(figsize=(8, 5))
# _{subscript} \beta^{superscript} \\hat{y}
plt.plot(y_hat, loss_y1, label="y=1: $-\\ln(f(x^i))$", color="blue")
plt.plot(y_hat, loss_y0, label="y=0: $-\\ln(1-f(x^i))$", color="red")
plt.xlabel("$f(x^i)$")
plt.ylabel("J")
plt.xlim(0, 1)
plt.ylim(0, 5)
plt.title("LOSS J -- $f(x^i)$")
plt.legend()
plt.grid()
plt.show()