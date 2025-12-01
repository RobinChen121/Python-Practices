"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/10/20 20:34
Description: lstm, criteria L1Loss() is better than MSELoss()


"""

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from statsmodels.datasets import get_rdataset


# -------------------------------
# 加载数据
# -------------------------------
dataset = get_rdataset("AirPassengers").data
raw_data = dataset["value"].values.astype(float)

# 归一化
data_min = raw_data.min()
data_max = raw_data.max()
data = (raw_data - data_min) / (data_max - data_min)


# -------------------------------
# 创建序列
# -------------------------------
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 12  # 用过去一年数据预测下一月
X, y = create_sequences(data, seq_length)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # 对于一维数据，增加维度
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)


# -------------------------------
# 定义 LSTM
# -------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)  # 线性输出

    def forward(self, x):
        out, _ = self.lstm(x)
        # out 的维度 (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]
        out = self.linear(out)
        return out


model = LSTMModel()
criterion = nn.L1Loss()  # nn.MSELoss()  #
# model.parameters() 包含模型训练的各项权重与偏置
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------------
# 训练
# -------------------------------
epochs = 300
for epoch in range(epochs):
    optimizer.zero_grad()  # 每次循环时梯度清零，不累加；模拟大 batch 时累加
    output = model(X)

    loss = criterion(output, y)
    loss.backward()
    optimizer.step()  # 更新参数
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# -------------------------------
# 预测
# -------------------------------
model.eval()  # 切换到评估模式
pred = model(X).detach().numpy()  # detach 分离张量

# -------------------------------
# 计算预测误差
# -------------------------------
y_true = y.detach().numpy() * (data_max - data_min) + data_min
y_pred = pred * (data_max - data_min) + data_min

# 平均绝对误差 MAE
mae = np.mean(np.abs(y_pred - y_true))

# 总绝对误差 SAE（sum of absolute errors）
sae = np.sum(np.abs(y_pred - y_true))

# （可选）均方根误差 RMSE
rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

print(f"平均绝对误差 (MAE): {mae:.6f}")
print(f"总绝对误差 (SAE): {sae:.6f}")
print(f"均方跟误差 (RMSE): {rmse:.6f}")

import matplotlib

matplotlib.use("TkAgg")  # 或者 "Qt5Agg"，具体取决于环境中装了哪个
import matplotlib.pyplot as plt

plt.plot(range(len(data)), data, label="True")
plt.plot(range(seq_length, len(data)), pred, label="Predicted")
plt.legend()
plt.show()
