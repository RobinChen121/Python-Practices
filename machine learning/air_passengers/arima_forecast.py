"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 17/11/2026, 22:42
@Desc: SARIMA (seasonal arima) forecasting for AirPassengers.
    Supports fixed-origin 12-step forecasting.
    Evaluation: MAE, MAPE, RMSE, MASE.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.datasets import get_rdataset
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

# -------------------------------
# 1. Configuration
# arima 预测所有test 的时间步，而不是像 lstm 那样指定预测步数
# -------------------------------
train_ratio = 0.8
seasonal_period = 12
p, d, q = 2, 1, 2
P, D, Q, s = 1, 1, 1, 12
order = (p, d, q)
s = seasonal_period
seasonal_order = (P, D, Q, s)

# -------------------------------
# 2. Load AirPassengers
# -------------------------------
df = get_rdataset("AirPassengers").data

raw_data = df["value"].to_numpy(dtype=float)

# adf 检验确定 d=1，季节性差分确定 D=1
from statsmodels.tsa.stattools import adfuller

diff_data = df["value"].diff(12).diff(1).dropna()
result = adfuller(diff_data)

print("ADF Statistic:", result[0])
print(
    "p-value:", result[1]
)  # p-value < 0.05：拒绝存在单位根的原假设，有证据支持序列平稳。

# 绘制 acf, pacf 图，确定 p, q, P, Q
# 这个图和刚才的adf检验仅供参考，实际仍然需要尝试多个值，比较预测误差
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(diff_data, lags=36, ax=axes[0])

plot_pacf(diff_data, lags=36, ax=axes[1], method="ywm")

plt.tight_layout()  # tight_layout() 函数用于自动调整子图参数，使之填充整个图像区域
plt.show()

# -------------------------------
# 3. Train / Test Split
# -------------------------------
train_len = int(len(raw_data) * train_ratio)

train_data = raw_data[:train_len]
test_data = raw_data[train_len:]

print("Training length:", len(train_data))
print("Testing length:", len(test_data))

# -------------------------------
# 4. Train SARIMA
# -------------------------------
model = SARIMAX(
    train_data,
    order=order,
    seasonal_order=seasonal_order,
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
)

model_fit = model.fit(disp=False)

# -------------------------------
# 5. Training Predictions
# -------------------------------
# In-sample one-step-ahead predictions.
# Skip early periods affected by initialization.
train_start = seasonal_period * D + d

train_result = model_fit.get_prediction(
    start=train_start, end=train_len - 1, dynamic=False
)

pred_train = np.asarray(train_result.predicted_mean)

# -------------------------------
# 6. Testing Predictions
# -------------------------------
# Forecast the entire test set.
# No actual test observations are used.

test_result = model_fit.get_forecast(steps=len(test_data))

pred_test = np.asarray(test_result.predicted_mean)

# 95% prediction intervals
conf_int = np.asarray(test_result.conf_int(alpha=0.05))

lower_bound = conf_int[:, 0]
upper_bound = conf_int[:, 1]

# -------------------------------
# 7. Evaluation Metrics
# -------------------------------
mae = mean_absolute_error(test_data, pred_test)
mape = mean_absolute_percentage_error(test_data, pred_test) * 100
rmse = np.sqrt(mean_squared_error(test_data, pred_test))

# Seasonal MASE
naive_mae = np.mean(
    np.abs(train_data[seasonal_period:] - train_data[:-seasonal_period])
)

mase = mae / naive_mae
print("\nTesting Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.4f}%")
print(f"MASE : {mase:.4f}")
print(f"RMSE : {rmse:.4f}")
# compute naive MASE using seasonal naive forecast
# Use observations from 12 months earlier
naive_pred = raw_data[train_len - seasonal_period : len(raw_data) - seasonal_period]
mae_baseline = mean_absolute_error(test_data, naive_pred)
mase_naive = mae_baseline / naive_mae
print(f"Seasonal Naive MASE: {mase_naive:.4f}")

# -------------------------------
# 8. Visualization
# -------------------------------
plt.figure(figsize=(12, 5))

# Actual observations
plt.plot(range(len(raw_data)), raw_data, label="Actual", color="blue")

# Training predictions
plt.plot(
    range(train_start, train_len), pred_train, label="Predicted (Train)", color="green"
)

# Testing predictions
plt.plot(
    range(train_len, len(raw_data)), pred_test, label="Predicted (Test)", color="red"
)

# 95% Prediction Interval
plt.fill_between(
    range(train_len, len(raw_data)),
    lower_bound,
    upper_bound,
    color="red",
    alpha=0.15,
    label="95% Prediction Interval",
)

# Training / testing split
plt.axvline(x=train_len, color="gray", linestyle="--", label="Train/Test Split")

plt.xlabel("Month Index")
plt.ylabel("Passengers")
plt.title("AirPassengers: SARIMA Forecast")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
