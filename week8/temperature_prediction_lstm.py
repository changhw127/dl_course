#  -*- utf-8 -*-
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from utlis.globals import Logger


class ClimateDataset(Dataset):
    def __init__(self, data, lookback=144, forecast_horizon=6):
        self.X, self.y = self.create_sequences(data, lookback, forecast_horizon)

    def create_sequences(self, data, lookback, forecast_horizon):
        X, y = [], []
        for i in range(len(data) - lookback - forecast_horizon):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback:i+lookback+forecast_horizon, 1])
        return torch.from_numpy(np.array(X)), torch.from_numpy(np.array(y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data():
    df = pd.read_csv('../data/course/jena_climate_2009_2016.csv')
    # Index(['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
    #        'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
    #        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
    #        'wd (deg)']

    features = [
                # 'Date Time',
                'p (mbar)', 'T (degC)', 'rh (%)', 'VPmax (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
                'H2OC (mmol/mol)', 'rho (g/m**3)',
                ]

    data = df[features].values.astype('float32')

    scaler = StandardScaler()
    scale_data = scaler.fit_transform(data).astype('float32')

    train_size = int(len(scale_data) * 0.7)
    val_size = int(len(scale_data) * 0.2)

    train_data = scale_data[:train_size]
    val_data = scale_data[train_size:train_size+val_size]
    test_data = scale_data[train_size+val_size:]

    lookback = 144  # 24小时的数据，10min间隔
    forest_horizon = 6  # 预测未来1h小时，6个时间步

    train_dataset = ClimateDataset(train_data, lookback, forest_horizon)
    val_dataset = ClimateDataset(val_data, lookback, forest_horizon)
    test_dataset = ClimateDataset(test_data, lookback, forest_horizon)

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    return features, train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 层
        out, (h_n, c_n) = self.lstm(x)
        # 获取最后一个时间步的输出
        out = out[:, -1, :]
        # 添加 Dropout
        out = self.dropout(out)
        # 全连接层
        out = self.fc(out)
        return out


def run():
    # lookback = 144  # 24小时的数据，10min间隔
    # forest_horizon = 6  # 预测未来1h小时，6个时间步
    features, train_data, val_data, test_data, train_dataloader, val_dataloader, test_dataloader = prepare_data()
    input_size = len(features)
    hidden_size = 128
    output_size = 6
    device = torch.device("mps")
    model = LSTMModel(input_size, hidden_size, output_size, num_layers=2)
    model.to(device)

    print(model)
    print(f'train_data:{len(train_data)}')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    epochs = 20
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        ix = 0
        total = len(train_dataloader)
        for inputs, targets in train_dataloader:
            ix += 1
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # 打印进度条
            Logger.progress_bar(ix, total)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # 绘制曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.savefig('temperature_prediction_lstm.png', dpi=100)
    plt.show()
    plt.close()


if __name__ == '__main__':
    run()

    """
LSTMModel(
  (lstm): LSTM(8, 64, num_layers=2, batch_first=True, dropout=0.2)
  (dropout): Dropout(p=0.2, inplace=False)
  (fc): Linear(in_features=64, out_features=6, bias=True)
)
Epoch 1/5, Train Loss: 0.0208, Val Loss: 0.0080
Epoch 2/5, Train Loss: 0.0142, Val Loss: 0.0070
Epoch 3/5, Train Loss: 0.0135, Val Loss: 0.0062
Epoch 4/5, Train Loss: 0.0130, Val Loss: 0.0058
Epoch 5/5, Train Loss: 0.0130, Val Loss: 0.0059

看到：欠拟合了，需要补充参数，这里先补充隐藏层到128
LSTMModel(
  (lstm): LSTM(8, 128, num_layers=2, batch_first=True, dropout=0.2)
  (dropout): Dropout(p=0.2, inplace=False)
  (fc): Linear(in_features=128, out_features=6, bias=True)
)
train_data:294385
Epoch 1/20, Train Loss: 0.0138, Val Loss: 0.0072
Epoch 2/20, Train Loss: 0.0102, Val Loss: 0.0070
Epoch 3/20, Train Loss: 0.0097, Val Loss: 0.0058
Epoch 4/20, Train Loss: 0.0095, Val Loss: 0.0051
Epoch 5/20, Train Loss: 0.0093, Val Loss: 0.0050
Epoch 6/20, Train Loss: 0.0094, Val Loss: 0.0050
Epoch 7/20, Train Loss: 0.0093, Val Loss: 0.0050
Epoch 8/20, Train Loss: 0.0091, Val Loss: 0.0049
Epoch 9/20, Train Loss: 0.0091, Val Loss: 0.0053
Epoch 10/20, Train Loss: 0.0090, Val Loss: 0.0060
Epoch 11/20, Train Loss: 0.0092, Val Loss: 0.0051
Epoch 12/20, Train Loss: 0.0092, Val Loss: 0.0049
Epoch 13/20, Train Loss: 0.0089, Val Loss: 0.0050
Epoch 14/20, Train Loss: 0.0089, Val Loss: 0.0052
Epoch 15/20, Train Loss: 0.0089, Val Loss: 0.0047
Epoch 16/20, Train Loss: 0.0091, Val Loss: 0.0051
Epoch 17/20, Train Loss: 0.0090, Val Loss: 0.0063
Epoch 18/20, Train Loss: 0.0110, Val Loss: 0.0080
Epoch 19/20, Train Loss: 0.0122, Val Loss: 0.0059
Epoch 20/20, Train Loss: 0.0105, Val Loss: 0.0053
    """