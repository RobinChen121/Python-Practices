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
# # fmt: off
# raw_data = np.array([
#     112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
#     115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
#     145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
#     171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
#     196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
#     204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
#     242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
#     284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
#     315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
#     340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
#     360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
#     417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
# ], dtype=float)
# #fmt on

dataset = get_rdataset("AirPassengers").data
raw_data = dataset["value"]

# 归一化
# data_min = raw_data.min()
# data_max = raw_data.max()
# data = -1 + 2 * (raw_data - data_min) / (data_max - data_min)
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(raw_data.to_numpy().reshape(-1, 1)) # scaler 只接受二维数组
data = data.flatten()

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
# 输入维度 [batch, seq_len, input_size]
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
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
        # input_size: the number of expected features in the input x
        # hidden_size: the number of features in the hidden stage h
        # num_layers: number of recurrent layers
        # 若 batch_first=True, lstm 输入数据的形状为 (N, L, H_in)，即 (batch_size, seq_length, input_size)
        # 否则，输入形状为 (seq_length, batch_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # D=2 if bidirectional otherwise =1
        # 若 batch_first=true, 每个时间步 h_t 形状一般(不考虑 proj_size)是 [num_layers * D, batch_size, hidden_size]
        # 若 batch_first=false, h_t 形状一般为 [num_layers * D, hidden_size]，即去掉 batch_size
        # 若 batch_first=true, 每个时间步 c_t 形状 [num_layers * D, batch_size, hidden_size]

        # 用最后那个 layer 的h_t 参与下面的映射
        self.linear = nn.Linear(
            hidden_size, output_size
        )  # 线性输出,输出维度 [batch, output_size]

    def forward(self, x):
        # 类中有 __call__() 函数，所以类可以直接调用
        # out 的形状一般(不考虑 proj_size)是 (batch_size, seq_size, hidden_size)
        # D=2 if bidirectional otherwise =1
        # 另一个输出 h_n 的形状是 (num_layers*D, batch_size, hidden_size)
        # 另一个输出 c_n 的形状是 (num_layers*D, batch_size, hidden_size)
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
model.train()  # 告诉模型处于训练模式
for epoch in range(epochs):
    # batch training
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch) # 相当于调用了里面的forward函数
        loss = criterion(output, y_batch)  # 这个loss 是 batch 的平均 loss
        loss.backward()  # 计算梯度
        optimizer.step()  # 根据梯度更新权重
        epoch_loss += (
            loss.item()
        )  # .item() 的核心作用：把单元素张量转换成普通 Python 数字
    epoch_loss /= len(train_loader)  # 平均loss
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.6f}")

# -------------------------------
# 预测
# -------------------------------
model.eval()  # 切换到评估模式
with torch.no_grad(): # 告诉 PyTorch，“接下来的计算不需要记录梯度（Gradients）”
    pred_train_norm = model(X_train)
    pred_test_norm = model(X_test)
    pred_train = scaler.inverse_transform(pred_train_norm.numpy())
    pred_test = scaler.inverse_transform(pred_test_norm.numpy())
    y_train_real = scaler.inverse_transform(y_train.numpy())
    y_test_real = scaler.inverse_transform(y_test.numpy())


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
