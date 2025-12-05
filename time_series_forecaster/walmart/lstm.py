"""
@Python version: 3.12
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 03/12/2025, 17:18
@Desc:

"""

import numpy as np
from read_data import read_data
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import optuna
from utils.get_pc_info import get_info

df = read_data()

# 选择一个 store+dept
store_id = 1
dept_id = 1
ts = df[(df.Store == store_id) & (df.Dept == dept_id)]

sales = ts["Weekly_Sales"].values.astype(np.float32)
is_holiday = ts["IsHoliday"].astype(float).values

# 需要标准化
# 0-1 变量不需要标准化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales.reshape(-1, 1))

# formulate a sequence
seq_len = 12  # 使用过去 12 周预测下一周


def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i: i + seq_len]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# data = np.column_stack((sales_scaled, is_holiday))
X, y = create_sequences(sales_scaled, seq_len)
X = torch.tensor(X)  # (batch, seq, feature)
y = torch.tensor(y)

dataset = torch.utils.data.TensorDataset(X, y)


# define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out


def run_model(batch_size, hidden_size, num_layers, bidirectional = False, learning_rate = 0.001):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = LSTMModel(input_size=X.shape[2], hidden_size=hidden_size, num_layers=num_layers,
                      bidirectional=bidirectional)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()  # 对于剧烈波动的数据，用 L1Loss 好

    # priority using GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # mac gpu
    else:
        device = torch.device("cpu")
    model.to(device)

    epochs = 30
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)  # 必须将输入数据和模型放在同一个 device
            batch_y = batch_y.to(device)
            # 这个循环 loader 里面的数据一个一个传进去
            optimizer.zero_grad()
            # lstm 的输入数据维度是 (batch_size, seq_size, input_size)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        # print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        X_device = X.to(device)
        y_device = y.to(device)
        pred = model(X_device)
        loss = criterion(pred, y_device)
        real_y = scaler.inverse_transform(y)
        real_pred = scaler.inverse_transform(pred.cpu().numpy())
        real_loss = criterion(torch.tensor(real_pred), torch.tensor(real_y))
    print(f"MAE: {loss.item():.4f}")
    print(f"real MAE: {real_loss.item():.4f}")
    return real_loss.item(), real_pred


def objective(trial):
    # 超参数采样
    hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64])
    num_layers = trial.suggest_int('num_layers', 1, 3)  # sample from 1 to 3
    # dropoutΩpout = trial.suggest_float('dropout', 0.0, 0.5)
    bidirectional = trial.suggest_categorical('bidirectional', [False])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)  # 1e-4 到 1e-2 的范围内用对数采样学习率
    batch_size = trial.suggest_categorical('batch_size', [1, 16, 32, 64])

    loss, _ = run_model(batch_size, hidden_size, num_layers, bidirectional, learning_rate)
    return loss


# 创建优化器
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)  # 20次尝试

print("best hyperparameters:", study.best_params)

import os

file_name = os.path.basename(__file__)
text_name = file_name.split(".py")[0]
text_name += '_record.txt'
with open(text_name, "a") as f:
    import platform
    import time

    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f.write("\n")
    f.write(f"date: {formatted_time}\n")
    f.write(f"file name: {file_name}\n")

    info = get_info()
    for k, v in info.items():
        f.write(f"{k}: {v}\n")
    if torch.cuda.is_available():
        f.write(f"GPU: {torch.cuda.is_available()}\n")
    else:
        f.write(f"GPU: {torch.backends.mps.is_available()}\n")
    f.write(f"Python version: {platform.python_version()}\n")
    f.write(f"best hyperparameters:\n{str(study.best_params)}\n")
    f.write(f"real MAE: {str(study.best_value)}\n")
    equals = "==" * 50
    f.write(f"{equals}\n")

_, real_pred = run_model(**study.best_params)
import matplotlib

matplotlib.use("Qt5Agg")  # 或者 "Qt5Agg"，具体取决于环境中装了哪个: conda install pyqt
import matplotlib.pyplot as plt

plt.plot(sales, label="real sales")
plt.plot(
    np.concatenate((np.array(sales[:seq_len]), real_pred.flatten())),
    label="predicted sales",
)
plt.legend()
plt.show()
