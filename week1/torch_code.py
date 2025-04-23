# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim


def print_gpu_memory():
    if torch.backends.mps.is_available():
        # MPS 内存统计（需 PyTorch 2.0+）
        allocated = torch.mps.current_allocated_memory() / 1024 ** 2  # MB
        reserved = torch.mps.driver_allocated_memory() / 1024 ** 2
        print(f"显存占用: {allocated:.2f} MB / 总保留: {reserved:.2f} MB")
    else:
        print("MPS 不可用")


def run_test():
    # 检查 MPS 是否可用
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建模型并移至 MPS
    model = nn.Linear(10, 1).to(device)

    # 生成数据并移至 MPS
    inputs = torch.randn(5, 10).to(device)
    labels = torch.randn(5, 1).to(device)

    # 训练循环
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        print_gpu_memory()


if __name__ == '__main__':
    run_test()
