import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from torch_geometric.transforms import NormalizeFeatures

# 加载数据集
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
data = dataset[0]  # 获取图数据

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, dataset.num_classes, heads=1)
        self.dropout = 0.6

    def forward(self, x, edge_index):
        # 第一层GAT (多头注意力)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 第二层GAT (单头输出)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 创建模型 (8头注意力)
model = GAT(hidden_channels=8, heads=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()

# 训练函数 (与GCN相同)
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 测试函数 (与GCN相同)
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

# 训练循环
for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 50 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
              f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# 最终测试
train_acc, val_acc, test_acc = test()
print(f'\nGAT Final Result: Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')