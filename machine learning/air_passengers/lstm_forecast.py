"""
Python version: 3.13
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/10/20 20:34
Description: lstm is slightly better than rnn;
forecasting the changes relative to the last observations is better than forecasting the absolute value;


"""

import torch
import torch.nn as nn
import numpy as np
from statsmodels.datasets import get_rdataset
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

# -------------------------------
# 加载数据
# -------------------------------
# # fmt: off # 是否开启 black 格式化
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

# configuration
torch.manual_seed(42)
# np.random.seed(42) # 并没有用 numpy生成随机数，这个设置可以省略

epoch_num = 300
seq_length = 12  # 用过去几个月的数据预测未来
pred_length = 1  # 预测未来几个月
season_length = 12

learning_rate = 0.01
batch_size = 24
hidden_size = 32
num_layers = 1

train_ratio = 0.8

dataset = get_rdataset("AirPassengers").data
raw_data = dataset["value"].to_numpy()
train_data = raw_data[: int(len(raw_data) * train_ratio)]

# 归一化 min, max scaling into [-1, 1]
# Fit on training observations
data_min = train_data.min()
data_max = train_data.max()
# Transform all data using training parameters
scaled_data = 2 * (raw_data - data_min) / (data_max - data_min) - 1


def inverse_transform(scaled_data):
    return (scaled_data + 1) * (data_max - data_min) / 2 + data_min


# -------------------------------
# 创建序列
# -------------------------------
def create_sequences(data, seq_length, pred_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        # Lstm 输入数据的 shape 是 [batch_size, seq_length, input_size]，input_size 是特征维度
        # 目前特征维度是一维的，shape [seq_length]，需要 unsqueeze(-1) 变为 shape [seq_length, 1]
        # unsqueeze(-1) 的作用是将 shape 从 [seq_length] 变为 [seq_length, 1]，即在最后一个维度增加一个维度
        x = torch.tensor(data[i: i + seq_length], dtype=torch.float32).unsqueeze(-1)  # shape [seq_length, 1]
        # lstm 的输出是 shape [batch, output_size]，output_size 是预测维度
        # 因为当前预测维度不是标量，所以不需要 unsqueeze(-1)来增加维度
        y = torch.tensor(data[i + seq_length: i + seq_length + pred_length], dtype=torch.float32)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)  # shape [N, seq_length, output_size], [N, output_size]


train_len = int(len(raw_data) * train_ratio)
# 分别生成训练与测试序列
X_train, y_train = create_sequences(
    scaled_data[:train_len],
    seq_length,
    pred_length
)
X_test, y_test = create_sequences(
    scaled_data[train_len - seq_length:],
    seq_length,
    pred_length
)
input_size = X_train.shape[-1]  # 输入特征维度

# 使用 batch size
# 若不使用，相当于 full batch
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # shuffle 表示打乱
    drop_last=False,  # 如果最后一个 batch 的样本数量不足 batch_size，就直接丢掉
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -------------------------------
# 定义 LSTM
# -------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=1, output_size=1):
        super().__init__()
        # 若 batch_first=True, lstm 输入数据的形状为 (batch_size, seq_length, input_size)
        # 否则，输入形状为 (seq_length, batch_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # D=2 if bidirectional otherwise =1
        # h_n.shape  # (num_layers * D, batch_size, hidden_size)
        # c_n.shape  # (num_layers * D, batch_size, hidden_size)

        # 用最后那个 layer 的h_t 参与下面的映射
        self.linear = nn.Linear(
            hidden_size, output_size
        )  # 线性输出,输出维度 [batch, output_size]

    def forward(self, x, h=None, c=None):
        # 若 batch_first=true, 每个时间步的输出形状是 (batch_size, seq_size, hidden_size*D)
        # D=2 if bidirectional otherwise =1
        # 若 batch_first=false, 输出形状是 (seq_size, batch_size, hidden_size*D)

        # self.lstm(x) 返回两个值：out 和 (h_n, c_n)
        # out 为所有时间步最后一层的隐藏状态 h，不包括 c
        # h_n 为所有层最后一个时间步的隐藏状态，c_n 为所有层最后一个时间步的细胞状态
        # out 的形状是 (batch_size, seq_len, hidden_size*D) 若 batch_first=true
        # h_n 的形状是 (num_layers*D, batch_size, hidden_size)
        # c_n 的形状是 (num_layers*D, batch_size, hidden_size)

        # 若样本假设相互独立，每次预测时都从初始状态开始
        # 否则，h, c 可以传入上一次的状态，用于预测下一个时间步
        if h is None or c is None:  # 第一次 forward 时，h, c 是 None，后续预测时会传入上一次的 h, c
            out, (h, c) = self.lstm(x)
        else:
            out, (h, c) = self.lstm(x, (h, c))

        # out, _ = self.lstm(
        #     x
        # )
        out = out[:, -1, :]

        # 等价于
        # _, out = self.lstm(
        #     x
        # )
        # out = out[0][-1]  # 取最后一个时间步的 h_n, out[0] 是 h_n, out[1] 是 c_n

        # LSTM 最后一个时间步的隐藏状态（Hidden State）可以通过一个全连接层 nn.Linear
        # 映射到任意指定维度的输出
        # out = self.linear(out) # 要通过linear函数把hidden state映射到output_size维度
        # return out

        # Predict changes relative to the last observation
        change = self.linear(out)  # shape [batch_size, pred_length]

        # baseline values
        # 0:1 会返回 shape [batch_size, seq_length, 1]，而 x[:, :, 0] 返回 shape [batch_size, seq_length]
        if pred_length == 1:
            baseline = x[:, -season_length]
        else:
            baseline = x[:, -season_length:, 0]

        return baseline + change


model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=pred_length)
criterion = nn.L1Loss()  # nn.MSELoss(),  L1Loss() 是 mae，MSELoss() 是 mse，mae 对异常值不敏感，mse 对异常值敏感
# model.parameters() 包含模型训练的各项权重与偏置
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 训练
# -------------------------------
epochs = epoch_num
model.train()  # 告诉模型处于训练模式
for epoch in range(epochs):
    # batch training
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)  # 相当于调用了里面的forward函数
        # PyTorch 中的 loss 函数可以自动计算一个 batch 内所有样本、所有预测时间步的平均绝对误差
        loss = criterion(output, y_batch)  # 这个 loss 是 batch 的平均 loss，
        loss.backward()  # 计算梯度
        optimizer.step()  # 根据梯度更新权重
        epoch_loss += (
                loss.item() * len(X_batch)
        )  # .item() 的核心作用：把单元素张量转换成普通 Python 数字
    epoch_loss /= len(train_dataset)  # train_loader 是batch的数量，不是样本总数
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.6f}")

# -------------------------------
# 预测
# -------------------------------
model.eval()  # 切换到评估模式
with torch.no_grad():  # 告诉 PyTorch，“接下来的计算不需要记录梯度（Gradients）”
    pred_train_norm = model(X_train)
    pred_test_norm = model(X_test)  # shape [N, pred_length, 1]
    pred_train = inverse_transform(pred_train_norm.numpy())
    pred_test = inverse_transform(pred_test_norm.numpy())
    y_train_real = inverse_transform(y_train.numpy())
    y_test_real = inverse_transform(y_test.numpy())

# -------------------------------
# 计算预测误差
# 这几个函数支持对二维数组计算误差
# -------------------------------
rmse_test = root_mean_squared_error(y_test_real, pred_test)
mae_test = mean_absolute_error(y_test_real, pred_test)
mape_test = mean_absolute_percentage_error(y_test_real, pred_test) * 100  # 转换为百分比

# MASE: use training data to calculate denominator
m = season_length  # AirPassengers annual seasonality
mae_naive = np.mean(
    np.abs(train_data[m:] - train_data[:-m])
)

mase_test = mae_test / mae_naive  # MASE (Mean Absolute Scaled Error)
print(f"测试集 MAE: {mae_test:.4f}")
print(f"测试集 MAPE: {mape_test:.2f}%")
print(f"测试集 RMSE: {rmse_test:.4f}")
print(f"测试集 MASE: {mase_test:.4f}")
# compute naive MASE using seasonal naive forecast
naive_pred = inverse_transform(
    X_test[:, :pred_length, 0].numpy()
)
mae_baseline = mean_absolute_error(
    y_test_real,
    naive_pred
)
mase_naive = mae_baseline / mae_naive
print(f"Seasonal Naive MASE: {mase_naive:.4f}")

# --------------------------
# 9. 可视化
# --------------------------

# -------------------------------
# Visualization
# -------------------------------
plt.figure(figsize=(12, 5))

# Actual observations
plt.plot(
    range(len(raw_data)),
    raw_data,
    label="Actual",
    color="blue"
)

# Training predictions (first forecast step)
# First step from each test window
train_pred = pred_train[:, 0]
# Add remaining steps from the last test window
test_pred = np.concatenate([
    train_pred,
    pred_train[-1, 1:]
])
plt.plot(
    range(seq_length, seq_length + len(test_pred)),
    test_pred,
    label="Predicted (Train)",
    color="green"
)

# Test predictions
# First step from each test window
test_pred = pred_test[:, 0]

# Add remaining steps from the last test window
test_pred = np.concatenate([
    test_pred,
    pred_test[-1, 1:]
])

plt.plot(
    range(train_len, train_len + len(test_pred)),
    test_pred,
    label="Predicted (Test)",
    color="red"
)

# Training / testing split
plt.axvline(
    x=train_len,
    color="gray",
    linestyle="--",
    label="Train/Test Split"
)

plt.xlabel("Month Index")
plt.ylabel("Passengers")
plt.title("AirPassengers: LSTM Direct Multi-step Forecast")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
