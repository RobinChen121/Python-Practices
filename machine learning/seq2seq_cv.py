"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/12/2 12:58
Description: implement seq2seq, adding month as the covariates


"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 读取 AirPassengers 数据
from statsmodels.datasets import get_rdataset

data = get_rdataset("AirPassengers").data
raw_data = data["value"]

# 添加月份信息
time_idx = np.arange(len(data))
month = time_idx % 12
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
# concatenate保持维度数量不变，指定维度变长
# stack 增加一个新的维度
covariates = np.stack([month_sin, month_cos], axis=1)

# 归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(raw_data.to_numpy().reshape(-1, 1)) # scaler 只接受二维数组
data = data.flatten()

# 构建序列
# shape 分别是
# X   = [N, encoder_length]
# y   = [N, decoder_length]
# cov = [N, encoder_length + decoder_length, cov_dim]
def create_sequences(data, covariates, encoder_length=12, decoder_length=1):
    X, y, cov = [], [], []
    for i in range(len(data) - encoder_length - decoder_length + 1):
        X.append(data[i : i + encoder_length])
        y.append(data[i + encoder_length : i + encoder_length + decoder_length])

        # covariates（关键：包含未来！）
        cov.append(
            covariates[
                i : i + encoder_length + decoder_length, :
            ]
        )
    return np.array(X), np.array(y), np.array(cov)


encoder_length = 12   # 用多少历史
decoder_length = 1    # 预测多少未来
batch_size = 12
hidden_dim = 32

X, y, cov = create_sequences(data, covariates, encoder_length, decoder_length)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
cov = torch.tensor(cov, dtype=torch.float32)

# 按时间切分 80/20
# --------------------------
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
cov_train, cov_test = cov[:split_idx], cov[split_idx:]

# 使用 batch size
# 若不使用，相当于 full batch
train_dataset = TensorDataset(X_train, y_train, cov_train)
test_dataset = TensorDataset(X_test, y_test, cov_test)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,  # 训练集通常 shuffle，但时间序列数据除外
    drop_last=False,  # 如果最后一个 batch 的样本数量不足 batch_size，就直接丢掉
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=hidden_dim, num_layers=1):
        super().__init__()
        cov_dim = 2 # 月份维度
        self.lstm = nn.LSTM(input_dim + cov_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x, cov):
        # 只取 encoder 部分 cov
        # 在 pytorch 中 size() 与 shape() 等价
        cov_enc = cov[:, :x.size(1), :]
        x = torch.cat([x, cov_enc], dim=-1)
        outputs, (h, c) = self.lstm(x)  # outputs 为每个所有时间步的隐藏状态
        return h, c


# decoder_length 预测未来多少个时间点
# output_dim 每一个时间点输出几个数
class Decoder(nn.Module):
    def __init__(self, decoder_length=1, hidden_dim=hidden_dim, output_dim=1, num_layers=1):
        super().__init__()
        self.decoder_length = decoder_length
        self.output_dim = output_dim
        cov_dim = 2 # 月份维度
        self.lstm = nn.LSTM(output_dim + cov_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(
            hidden_dim, output_dim
        )  # nn.Linear 对输入张量的最后一维做线性变换，hidden_dim, output_dim 分别是输入数据
        # 和输出数据的最后一个维度
        # 改为 self.linear 也可以，则下面 forward 中也得叫 self.linear

    def forward(self, h, c, cov):
        # decoder 输入初始化：用 0 向量作为起点
        # h 的形状: [num_layers * num_directions, batch_size, hidden_dim]
        # PyTorch 的 LSTM 输入必须是 3 维张量，格式是：
        # [batch_size, encoder_length, input_size]
        # input_size 是 the number of expected features in the input x
        # autoregressive 滚动
        batch_size = h.size(1)
        decoder_input = torch.zeros(batch_size, 1, 1, device=h.device) # 中间是 1 表示预测未来 1 个时间步

        outputs = []
        for t in range(self.decoder_length):
            # 取未来 covariate
            cov_t = cov[:, -(self.decoder_length) + t : -(self.decoder_length) + t + 1, :]
            lstm_input = torch.cat([decoder_input, cov_t], dim=-1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            pred = self.fc(out)

            outputs.append(pred)
            decoder_input = pred # 下一个时间步用前一步预测

        # for _ in range(self.decoder_length):
        #     # 第一个输入为 decoder_input
        #     out, (h, c) = self.lstm(decoder_input, (h, c))
        #     pred = self.fc(out)  #
        #     outputs.append(pred)
        #     decoder_input = pred  # 下一个时间步用前一步预测

        # 将这个list的元素拼接，不增加新的维度，维度为 [batch, decoder_length, 1]
        # 与 stack 不一样，stack 会增加维度
        outputs = torch.cat(outputs, dim=1)
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(decoder_length=decoder_length)

    def forward(self, x, cov):
        h, c = self.encoder(x, cov)
        out = self.decoder(h, c, cov)
        return out


model = Seq2Seq()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 300

model.train()  # 告诉模型处于训练模式
for epoch in range(epochs):
    # batch training
    epoch_loss = 0.0
    for X_batch, y_batch, cov_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch, cov_batch) # 相当于调用了里面的forward函数
        loss = criterion(output, y_batch)  # 这个loss 是 batch 的平均 loss
        loss.backward()  # 计算梯度
        optimizer.step()  # 根据梯度更新权重
        epoch_loss += (
            loss.item()
        )  # .item() 的核心作用：把单元素张量转换成普通 Python 数字
    epoch_loss /= len(train_loader)  # 平均loss
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.6f}")

model.eval()  # 切换到评估模式
with torch.no_grad(): # 告诉 PyTorch，“接下来的计算不需要记录梯度（Gradients）”
    pred_train_norm = model(X_train, cov_train)
    pred_test_norm = model(X_test, cov_test)
    pred_train_np = pred_train_norm.squeeze(-1).numpy()   # [N, decoder_length]
    pred_test_np = pred_test_norm.squeeze(-1).numpy()

    y_train_np = y_train.squeeze(-1).numpy()
    y_test_np = y_test.squeeze(-1).numpy()

    # reshape 成二维给 scaler
    # reshape 中的 -1 表示让numpy推理这个维度
    pred_train = scaler.inverse_transform(pred_train_np.reshape(-1, 1)).reshape(pred_train_np.shape)
    pred_test = scaler.inverse_transform(pred_test_np.reshape(-1, 1)).reshape(pred_test_np.shape)

    y_train_real = scaler.inverse_transform(y_train_np.reshape(-1, 1)).reshape(y_train_np.shape)
    y_test_real = scaler.inverse_transform(y_test_np.reshape(-1, 1)).reshape(y_test_np.shape)


# --------------------------
# 8. 计算 MSE
# --------------------------
# sklearn 会自动 flatten
mse_train = mean_squared_error(y_train_real, pred_train)
mse_test = mean_squared_error(y_test_real, pred_test)
# 将训练集和测试集按顺序拼接
y_total = np.concatenate((y_train_real, y_test_real))
pred_total = np.concatenate((pred_train, pred_test))

# 计算整个数据集的 MSE
mse_total = mean_squared_error(y_total, pred_total)
mse = np.mean((y_train_real - pred_train) ** 2) # y_train_real 的shape是 [N, decoder_length]
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
pred_train_1step = pred_train[:, 0]
pred_test_1step = pred_test[:, 0]
plt.plot(
    range(encoder_length, encoder_length + len(pred_train_1step)),
    pred_train_1step,
    label="Predicted (Train)",
)


# 绘制测试集预测（对应后 20%）
plt.plot(
    range(encoder_length + split_idx, encoder_length + split_idx + len(pred_test_1step)),
    pred_test_1step,
    label="Predicted (Test)",
    color="red",
)

# 分割线
plt.axvline(
    x=encoder_length + split_idx, color="gray", linestyle="--", label="Train/Test Split"
)

plt.xlabel("Month Index")
plt.ylabel("Passengers")
plt.title("AirPassengers Prediction using Seq2Seq (Train + Test)")
plt.legend()
plt.show()
