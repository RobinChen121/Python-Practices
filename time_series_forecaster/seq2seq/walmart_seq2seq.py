"""
@Python version: 3.12
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 27/11/2025, 23:50
@Desc:

"""

import pandas as pd
import numpy as np
import os
import sys

file_path = ""
if sys.platform != "win32":
    file_path = "/Users/zhenchen/Documents/machine learning data/walmart-recruiting-store-sales-forecasting"

# 读取数据
sales = pd.read_csv(os.path.join(file_path, "train.csv"), parse_dates=["Date"])
features = pd.read_csv(os.path.join(file_path, "features.csv"), parse_dates=["Date"])
stores = pd.read_csv(os.path.join(file_path, "stores.csv"))

# 合并数据
df = pd.merge(sales, features, on=["Store", "Date", "IsHoliday"])
df = pd.merge(df, stores, on="Store")

# 按时间排序
df = df.sort_values(["Store", "Dept", "Date"])

# 提取时间特征
df["weekofyear"] = df["Date"].dt.isocalendar().week
df["month"] = df["Date"].dt.month
# df["dayofweek"] = df["Date"].dt.dayofweek

# 将 IsHoliday 转成 int
df["IsHoliday"] = df["IsHoliday"].astype(int)

# 滞后特征示例
df["lag_1"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)
df["lag_4"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(4)
# df["lag_52"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(52)

# 删除 NaN 行（滞后引入的）
df = df.dropna()

from sklearn.preprocessing import StandardScaler

# 特征列`
feature_cols = ["weekofyear", "month", "IsHoliday", "lag_1", "lag_4"]  # “dayofweek"
target_col = "Weekly_Sales"

# 标准化特征
# 对输入特征 x 标准化，但对 y 不需要
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])


# 生成序列
# 根据过去 8 周预测未来 4 周
def create_sequences(data, input_len=8, output_len=4):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[feature_cols].iloc[i : i + input_len].values)
        y.append(
            data[target_col].iloc[i + input_len : i + input_len + output_len].values
        )
    return np.array(X), np.array(y)


X, y = create_sequences(df, input_len=8, output_len=4)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 数据转换为 tensor
# PyTorch 的神经网络模块（如 nn.LSTM, nn.Linear）只能接受 torch.Tensor 作为输入
# 普通的 numpy 数组或 Python list 不能直接送入网络
# 因为 PyTorch 的 Tensor 内部是 GPU/CPU 上的连续内存块，可以高效计算梯度
X_tensor = torch.Tensor(X)
y_tensor = torch.Tensor(y)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


# Seq2Seq 模型定义
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_len):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.output_len = output_len

    def forward(self, x):
        # Encoder
        _, (hidden, cell) = self.encoder(x)

        # Decoder input 初始化为最后一条输入
        decoder_input = x[:, -1, :].unsqueeze(1)
        outputs = []

        for _ in range(self.output_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = torch.cat(
                [decoder_input[:, :, :-1], pred], dim=2
            )  # 更新 decoder 输入

        return torch.cat(outputs, dim=1)


# 模型实例
input_size = len(feature_cols)
hidden_size = 64
output_len = 4

model = Seq2Seq(input_size, hidden_size, output_len)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred.squeeze(-1), yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    sample_input = X_tensor[:5]
    pred = model(sample_input)
    print(pred.shape)  # [5, output_len, 1]
