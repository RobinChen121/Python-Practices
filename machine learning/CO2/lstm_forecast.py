"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 18/11/2026, 11:30
@Desc: LSTM for Weekly CO2 forecasting.

"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

# -------------------------------
# 1. Configuration
# -------------------------------
seed = 42
torch.manual_seed(seed)
# np.random.seed(42)

epoch_num = 150

seq_length = 52
pred_length = 1  # 1: one-step; 12: multi-step

learning_rate = 0.001
batch_size = 64

hidden_size = 64
num_layers = 2

train_ratio = 0.8
seasonal_period = 52

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():  # for Apple Silicon (M1/M2...) devices
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# device = torch.device("cpu")

# -------------------------------
# 2. Load Weekly CO2
# -------------------------------
df = sm.datasets.co2.load_pandas().data

# Regular weekly observations and missing values
# 将 CO₂ 时间序列数据按周（每周六为结束日）重新采样合并，计算每周的平均值，向前补充缺失值
# 最后将最开头的缺失值（若有）删除，得到完整的时间序列数据
data = df["co2"].resample("W-SAT").mean().ffill().dropna()

raw_data = data.to_numpy(dtype=np.float32)

print("Data length:", len(raw_data))
print("First date:", data.index[0])
print("Last date:", data.index[-1])

# -------------------------------
# 3. Train / Test Split
# -------------------------------
train_len = int(len(raw_data) * train_ratio)

train_data = raw_data[:train_len]
test_data = raw_data[train_len:]

print("Training length:", len(train_data))
print("Testing length:", len(test_data))

# -------------------------------
# 4. Data Normalization
# -------------------------------
# Fit normalization on training data only
data_mean = train_data.mean()
data_std = train_data.std()

if data_std == 0:
    raise ValueError("Training std cannot be zero.")

scaled_data = (raw_data - data_mean) / data_std


def inverse_transform(x):
    return x * data_std + data_mean


# -------------------------------
# 5. Create Sequences
# -------------------------------
def create_sequences(data, seq_length, pred_length):
    X, y = [], []

    for i in range(len(data) - seq_length - pred_length + 1):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length : i + seq_length + pred_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return (torch.from_numpy(X).unsqueeze(-1), torch.from_numpy(y))


# Training sequences:
# Each target is completely within training data.
X_train, y_train = create_sequences(scaled_data[:train_len], seq_length, pred_length)

# Testing sequences:
# Include the last seq_length training observations
# so the first test target starts at train_len.
X_test, y_test = create_sequences(
    scaled_data[train_len - seq_length :], seq_length, pred_length
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

train_dataset = TensorDataset(X_train, y_train)
generator = torch.Generator()
generator.manual_seed(seed)  # 设置随机种子，保证每次运行结果一致
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, generator=generator
)


# -------------------------------
# 6. LSTM Model
# -------------------------------
class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, pred_length):
        super().__init__()

        # 这里 dropout=0.1 表示训练期间，层间输出 h 中的元素以 10% 的概率被随机置为零，以减轻过拟合
        # 维度并不变，h中的元素只是随机置为零，其他元素保持不变
        # h 作为下一层的输入
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,  #
        )

        self.linear = nn.Linear(hidden_size, pred_length)

    def forward(self, x):

        # out: (batch, seq_length, hidden_size)
        out, _ = self.lstm(x)

        # Last time step
        out = out[:, -1, :]

        # Direct multi-step prediction
        out = self.linear(out)
        baseline = x[:, -pred_length:, 0]

        return out + baseline  # Add baseline to the predicted values


model = LSTMModel(
    input_size=1,
    hidden_size=hidden_size,
    num_layers=num_layers,
    pred_length=pred_length,
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 7. Training
# -------------------------------
loss_history = []

for epoch in range(epoch_num):

    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(train_dataset)
    loss_history.append(avg_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epoch_num}, " f"Loss: {avg_loss:.6f}")

# -------------------------------
# 8. Prediction
# -------------------------------
model.eval()

with torch.inference_mode():
    pred_train = model(X_train.to(device)).cpu().numpy()
    pred_test = model(X_test.to(device)).cpu().numpy()

# Inverse transform predictions
pred_train = inverse_transform(pred_train)
pred_test = inverse_transform(pred_test)

# Actual values
y_train_real = inverse_transform(y_train.numpy())
y_test_real = inverse_transform(y_test.numpy())

# -------------------------------
# 9. Evaluation Metrics
# -------------------------------
# Evaluate every forecast horizon across all windows.
# This means overlapping target dates may be counted
# multiple times (at different forecast horizons).

y_true = y_test_real.flatten()
y_pred = pred_test.flatten()

mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
rmse = root_mean_squared_error(y_true, y_pred)

# Seasonal naive MAE on training data
mae_naive = mean_absolute_error(
    train_data[seasonal_period:], train_data[:-seasonal_period]
)

mase = mae / mae_naive

# -------------------------------
# 10. Seasonal Naive Baseline
# -------------------------------
# For each forecast target, use the observation
# from 52 weeks earlier.
#
# The first target window starts at train_len.
naive_pred = np.resize(train_data[-seasonal_period:], len(test_data))
mae_baseline = mean_absolute_error(y_true, naive_pred.flatten())

mase_naive = mae_baseline / mae_naive

print("\nTesting Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.4f}%")
print(f"RMSE : {rmse:.4f}")
print(f"MASE : {mase:.4f}")
print(f"Seasonal Naive MASE: {mase_naive:.4f}")

# -------------------------------
# 11. Visualization
# -------------------------------
plt.figure(figsize=(12, 5))

# Actual observations
plt.plot(range(len(raw_data)), raw_data, label="Actual", color="blue")

# Training predictions:
# First forecast step from each training window
train_plot = pred_train[:, 0]

# Add remaining steps from the last window
if pred_length > 1:
    train_plot = np.concatenate([train_plot, pred_train[-1, 1:]])

plt.plot(
    range(seq_length, seq_length + len(train_plot)),
    train_plot,
    label="Predicted (Train)",
    color="green",
)

# Testing predictions:
# First forecast step from each testing window
test_plot = pred_test[:, 0]

# Add remaining steps from the last window
if pred_length > 1:
    test_plot = np.concatenate([test_plot, pred_test[-1, 1:]])

plt.plot(
    range(train_len, train_len + len(test_plot)),
    test_plot,
    label="Predicted (Test)",
    color="red",
)

# Training / testing split
plt.axvline(x=train_len, color="gray", linestyle="--", label="Train/Test Split")

plt.xlabel("Week Index")
plt.ylabel("CO2 Concentration (ppm)")
plt.title("Weekly CO2: LSTM Direct Multi-step Forecast")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# 12. Training Loss
# -------------------------------
plt.figure(figsize=(10, 4))

plt.plot(loss_history, label="Training Loss", color="blue")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM Training Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# 13. Test Period Visualization
# -------------------------------
plt.figure(figsize=(12, 5))

plt.plot(
    range(train_len, len(raw_data)), test_data, label="Actual (Test)", color="blue"
)

plt.plot(
    range(train_len, len(raw_data)), test_plot, label="Predicted (Test)", color="red"
)

plt.xlabel("Week Index")
plt.ylabel("CO2 Concentration (ppm)")
plt.title("Weekly CO2: LSTM Testing Forecast")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
