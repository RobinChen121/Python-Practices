"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 25/10/2025, 14:01
@Desc    : vanilla rnn

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# --------------------------
# 1. 数据准备
# --------------------------
# fmt: off
raw_data = np.array([
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
    115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
    145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
    171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
    196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
    204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
    242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
    284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
    315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
    340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
    360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
    417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
], dtype=float)
# fmt: on

data_min = raw_data.min()
data_max = raw_data.max()
data = -1 + 2 * (raw_data - data_min) / (data_max - data_min)
data = torch.FloatTensor(data).unsqueeze(1)  # shape (144,1)


# --------------------------
# 2. 构建序列
# --------------------------
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i : i + seq_len]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


seq_len = 12
X_all, y_all = create_sequences(data, seq_len)

# --------------------------
# 3. 按时间切分 80/20
# --------------------------
split_idx = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split_idx], X_all[split_idx:]
y_train, y_test = y_all[:split_idx], y_all[split_idx:]

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")


# --------------------------
# 4. 定义原生 RNN 模型
# --------------------------
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = SimpleRNN(input_size=1, hidden_size=16, output_size=1)

# --------------------------
# 5. 损失函数 & 优化器
# --------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# --------------------------
# 6. 训练
# --------------------------
epochs = 300
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss.item():.6f}")

# --------------------------
# 7. 预测（训练集 + 测试集）
# --------------------------
model.eval()
with torch.no_grad():
    pred_train_norm = model(X_train)
    pred_test_norm = model(X_test)
    pred_train = (pred_train_norm.numpy() + 1) * (data_max - data_min) / 2 + data_min
    pred_test = (pred_test_norm.numpy() + 1) * (data_max - data_min) / 2 + data_min
    y_train_real = (y_train.numpy() + 1) * (data_max - data_min) / 2 + data_min
    y_test_real = (y_test.numpy() + 1) * (data_max - data_min) / 2 + data_min

# --------------------------
# 8. 计算 MSE
# --------------------------
mse_train = mean_squared_error(y_train_real, pred_train)
mse_test = mean_squared_error(y_test_real, pred_test)
# 将训练集和测试集按顺序拼接
y_total = np.concatenate((y_train_real, y_test_real))
pred_total = np.concatenate((pred_train, pred_test))

# 计算整个数据集的 MSE
mse_total = mean_squared_error(y_total, pred_total)
mse = 0
for a, b in zip(y_train_real, pred_train):
    print(a, b)
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
    range(seq_len, seq_len + len(pred_train)),
    pred_train,
    label="Predicted (Train)",
    color="green",
)

# 绘制测试集预测（对应后 20%）
plt.plot(
    range(seq_len + split_idx, seq_len + split_idx + len(pred_test)),
    pred_test,
    label="Predicted (Test)",
    color="red",
)

# 分割线
plt.axvline(
    x=seq_len + split_idx, color="gray", linestyle="--", label="Train/Test Split"
)

plt.xlabel("Month Index")
plt.ylabel("Passengers")
plt.title("AirPassengers Prediction using Simple RNN (Train + Test)")
plt.legend()
plt.show()
