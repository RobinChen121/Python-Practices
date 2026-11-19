"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 18/11/2026, 16:07
@Desc: using LSTM combined with maximum likelihood estimation to forecast AirPassengers dataset,
 with Monte Carlo simulation to generate multiple future paths and prediction intervals

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from statsmodels.datasets import get_rdataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

# --------------------------------
# 1. Configuration
# --------------------------------
torch.manual_seed(42)
np.random.seed(42)

epochs = 300
seq_length = 12
train_ratio = 0.8
season_length = 12  # monthly data, so season length is 12 months

hidden_size = 32
num_layers = 1
batch_size = 24
learning_rate = 0.01

num_scenarios = 1000

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用 CPU 进行训练和预测
device = torch.device("cpu")

# --------------------------------
# 2. Load data
# --------------------------------
df = get_rdataset("AirPassengers").data
data = df["value"].to_numpy(dtype=np.float32)
dates = pd.date_range(start="1949-01-01", periods=len(data), freq="MS")

train_size = int(len(data) * train_ratio)
train_data = data[:train_size]
test_data = data[train_size:]

# Fit scaler using training data only
scaler = StandardScaler()

train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()


# --------------------------------
# 3. Create training sequences
# --------------------------------
def create_sequences(data, seq_length):
    X, y = [], []

    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])

    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    return X, y


X_train, y_train = create_sequences(train_scaled, seq_length)

train_loader = DataLoader(
    TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
)


# --------------------------------
# 4. Gaussian LSTM
# --------------------------------
class GaussianLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mean_layer = nn.Linear(hidden_size, 1)
        self.std_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)

        h = out[:, -1, :]

        mu = (
            self.mean_layer(h) + x[:, -season_length:, 0:1]
        )  # add last observed value to mean prediction
        # softplus ensures that the standard deviation is positive, its formula is log(1 + exp(x)), which is always positive
        sigma = F.softplus(self.std_layer(h)) + 1e-6
        return mu, sigma


model = GaussianLSTM(hidden_size, num_layers).to(device)


# --------------------------------
# 5. Negative log likelihood
# --------------------------------
def gaussian_nll(y, mu, sigma):
    distribution = torch.distributions.Normal(loc=mu, scale=sigma)
    # 因为 y 是一个 batch 的数据，所以需要对每个样本的负对数似然求平均或加和，
    # 但一般取平均，因为 batch size 可能不一样，取平均可以使得 loss 的 scale 不受 batch size 影响
    return -distribution.log_prob(y).mean()


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --------------------------------
# 6. Training
# --------------------------------
loss_history = []

for epoch in range(epochs):
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        mu, sigma = model(X_batch)

        loss = gaussian_nll(y_batch, mu, sigma)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    epoch_loss = total_loss / len(X_train)
    loss_history.append(epoch_loss)

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:3d} | " f"Negative Log-Likelihood: {epoch_loss:.4f}")

# --------------------------------
# 7. Monte Carlo forecasting
# --------------------------------
model.eval()

pred_length = len(test_data)

# Last 12 months of training data
initial_window = torch.tensor(
    train_scaled[-seq_length:], dtype=torch.float32, device=device
).reshape(1, seq_length, 1)

# Replicate initial window for all paths
windows = initial_window.repeat(num_scenarios, 1, 1)

scenarios_scaled = []
with torch.inference_mode():

    for t in range(pred_length):

        # Conditional Gaussian parameters
        mu, sigma = model(windows)

        # Sample next observation
        sampled_y = torch.normal(mu, sigma)
        # sampled_y 的 shape 是 [num_scenarios, 1]，需要 squeeze(-1) 去掉最后一维
        scenarios_scaled.append(sampled_y.squeeze(-1).cpu().numpy())

        # Update each scenario's window
        windows = torch.cat([windows[:, 1:, :], sampled_y.unsqueeze(1)], dim=1)

scenarios_scaled = np.stack(scenarios_scaled, axis=1)

# Inverse scaling
scenarios = scaler.inverse_transform(scenarios_scaled.reshape(-1, 1)).reshape(
    num_scenarios, pred_length
)

# --------------------------------
# 8. Forecast distribution summary
# --------------------------------
prediction_mean = scenarios.mean(axis=0)
lower_95 = np.quantile(scenarios, 0.025, axis=0)
upper_95 = np.quantile(scenarios, 0.975, axis=0)

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

# Actual values in testing period
y_true = test_data

# LSTM Monte Carlo prediction mean
y_pred = prediction_mean

# --------------------------------
# 1. MAE, MAPE, RMSE
# --------------------------------
mae = mean_absolute_error(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred) * 100
rmse = root_mean_squared_error(y_true, y_pred)

# --------------------------------
# 2. MASE denominator
# --------------------------------
# One-step naive forecast error
# calculated using training data

naive_train_mae = np.mean(
    np.abs(train_data[season_length:] - train_data[:-season_length])
)

# LSTM MASE
mase = mae / naive_train_mae

# --------------------------------
# 3. Naive forecast
# --------------------------------
# Recursive multi-step naive forecast:
# all future values equal last training observation

naive_prediction = data[len(train_data) - season_length : len(data) - season_length]
naive_mae = mean_absolute_error(y_true, naive_prediction)
naive_mase = naive_mae / naive_train_mae

print(f"测试集 MAE: {mae:.4f}")
print(f"测试集 MAPE: {mape:.2f}%")
print(f"测试集 RMSE: {rmse:.4f}")
print(f"测试集 MASE: {mase:.4f}")
print(f"Seasonal Naive MASE: {naive_mase:.4f}")

# --------------------------------
# 9. Visualisation
# --------------------------------
test_dates = dates[train_size:]

plt.figure(figsize=(13, 6))

plt.plot(dates[:train_size], train_data, label="Training Data", color="black")

plt.plot(test_dates, test_data, label="Actual Test Data", color="blue")

# Show 20 sampled future paths
for i in range(20):
    plt.plot(test_dates, scenarios[i], color="yellow", alpha=0.15, linewidth=0.8)

plt.plot(
    test_dates, prediction_mean, color="red", linewidth=2, label="Monte Carlo Mean"
)

plt.fill_between(
    test_dates,
    lower_95,
    upper_95,
    color="red",
    alpha=0.2,
    label="95% Prediction Interval",
)

plt.axvline(dates[train_size], color="gray", linestyle="--")

plt.title("AirPassengers: Gaussian LSTM Forecast")

plt.xlabel("Date")
plt.ylabel("Monthly Passengers (Thousands)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# # --------------------------------
# # 10. Training loss
# # --------------------------------
# plt.figure(figsize=(9, 4))
# plt.plot(loss_history)
# plt.xlabel("Epoch")
# plt.ylabel("Gaussian Negative Log Likelihood")
# plt.title("Training Loss")
# plt.grid(alpha=0.3)
# plt.tight_layout()
# plt.show()
#
# # --------------------------------
# # 11. Forecast results
# # --------------------------------
# results = pd.DataFrame(
#     {
#         "Actual": test_data,
#         "Forecast_Mean": prediction_mean,
#         "Lower_95": lower_95,
#         "Upper_95": upper_95,
#     },
#     index=test_dates,
# )
#
# print(results.round(2))
