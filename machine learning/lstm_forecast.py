"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/10/20 20:34
Description: lstm, criteria L1Loss() is better than MSELoss()


"""

import torch
import torch.nn as nn
import numpy as np
from statsmodels.datasets import get_rdataset
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 加载数据
# -------------------------------
dataset = get_rdataset("AirPassengers").data
raw_data = dataset["value"].values.astype(float)

# 归一化
# data_min = raw_data.min()
# data_max = raw_data.max()
# data = -1 + 2 * (raw_data - data_min) / (data_max - data_min)
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(raw_data)


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
batch_size = 24
X, y = create_sequences(data, seq_length)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # 对于一维数据，增加维度
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# 按时间切分 80/20
# --------------------------
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 使用 batch size
# 若不使用，相当于 full batch
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,  # 训练集通常 shuffle，但时间序列数据除外
    drop_last=False,  # 如果最后一个 batch 的样本数量不足 batch_size，就直接丢掉
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -------------------------------
# 定义 LSTM
# -------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 把 LSTM 的 hidden state 映射到输出空间
        # 每个时间步 h_t 形状 [num_layers, batch, hidden_size]
        self.linear = nn.Linear(
            hidden_size, output_size
        )  # 线性输出,输出维度 [batch, output_size]

    def forward(self, x):
        # 类中有 __call__() 函数，所以类可以直接调用
        # out 的形状是 (batch_size, seq_size, hidden_size)
        out, _ = self.lstm(
            x
        )  # # 返回所有时间步的隐藏状态及最后一个时间步的（h_n, c_m）
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
    # batch training
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)  # 这个loss 是 batch 的平均 loss
        loss.backward()  # 计算梯度
        optimizer.step()  # 根据梯度更新权重
        epoch_loss += (
            loss.item()
        )  # .item() 的核心作用：把单元素张量转换成普通 Python 数字
    epoch_loss /= len(train_loader)  # 平均每个 batch 的 loss
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.6f}")

# -------------------------------
# 预测
# -------------------------------
model.eval()  # 切换到评估模式
with torch.no_grad():
    pred_train_norm = model(X_train)
    pred_test_norm = model(X_test)
    pred_train = (
        scaler.inverse_transform(pred_train_norm)(pred_train_norm.numpy() + 1)
        * (data_max - data_min)
        / 2
        + data_min
    )
    pred_test = (pred_test_norm.numpy() + 1) * (data_max - data_min) / 2 + data_min
    y_train_real = (y_train.numpy() + 1) * (data_max - data_min) / 2 + data_min
    y_test_real = (y_test.numpy() + 1) * (data_max - data_min) / 2 + data_min


# -------------------------------
# 计算预测误差
# -------------------------------
mse_train = mean_squared_error(y_train_real, pred_train)
mse_test = mean_squared_error(y_test_real, pred_test)
# 将训练集和测试集按顺序拼接
y_total = np.concatenate((y_train_real, y_test_real))
pred_total = np.concatenate((pred_train, pred_test))

# 计算整个数据集的 MSE
mse_total = mean_squared_error(y_total, pred_total)
mse = 0
for a, b in zip(y_train_real, pred_train):
    # print(a, b)
    mse += (a.item() - b.item()) ** 2
# for (a, b) in zip(y_test_real, pred_test):
#     print(a, b)
print(f"\n训练集 RMSE: {np.sqrt(mse_train):.4f}")
print(f"测试集 RMSE: {np.sqrt(mse_test):.4f}")
print(f"总 RMSE: {np.sqrt(mse_total):.4f}")

# --------------------------
# 9. 可视化
# --------------------------
import matplotlib

matplotlib.use("TkAgg")  # 或者 "Qt5Agg"，具体取决于你环境中装了哪个
plt.figure()
plt.plot(range(len(raw_data)), raw_data, label="Actual", color="blue")

# 绘制训练集预测（对应前 80%）
plt.plot(
    range(seq_length, seq_length + len(pred_train)),
    pred_train,
    label="Predicted (Train)",
    color="green",
)

# 绘制测试集预测（对应后 20%）
plt.plot(
    range(seq_length + split_idx, seq_length + split_idx + len(pred_test)),
    pred_test,
    label="Predicted (Test)",
    color="red",
)

# 分割线
plt.axvline(
    x=seq_length + split_idx, color="gray", linestyle="--", label="Train/Test Split"
)

plt.xlabel("Month Index")
plt.ylabel("Passengers")
plt.title("AirPassengers Prediction using LSTM (Train + Test)")
plt.legend()
plt.show()
