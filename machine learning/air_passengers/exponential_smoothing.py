"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 18/11/2026, 00:36
@Desc: Holt-Winters Seasonal Exponential Smoothing

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.datasets import get_rdataset
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

# -------------------------------
# 1. Configuration
# -------------------------------
train_ratio = 0.8
seasonal_period = 12

trend_type = "mul"
seasonal_type = "mul"

# -------------------------------
# 2. Load AirPassengers
# -------------------------------
df = get_rdataset("AirPassengers").data

raw_data = df["value"].to_numpy(dtype=float)

# -------------------------------
# 3. Train / Test Split
# -------------------------------
train_len = int(len(raw_data) * train_ratio)

train_data = raw_data[:train_len]
test_data = raw_data[train_len:]

print("Training samples:", len(train_data))
print("Testing samples:", len(test_data))

# -------------------------------
# 4. Holt-Winters Model
# -------------------------------
model = ExponentialSmoothing(
    train_data,
    trend=trend_type,
    seasonal=seasonal_type,
    seasonal_periods=seasonal_period,
    damped_trend=False,
    initialization_method="estimated",
)

# 通过 L-BFGS 算法优化平滑参数，最小化训练集的残差平方和
model_fit = model.fit(
    optimized=True
)  # model.fit(optimized=True) 会根据训练数据自动优化平滑参数及相关初始状态。

print("\nModel Parameters:")
print(model_fit.params)

# -------------------------------
# 5. Training Predictions
# -------------------------------
# In-sample fitted values
pred_train = np.asarray(model_fit.fittedvalues)

# -------------------------------
# 6. Testing Predictions
# -------------------------------
# Fixed-origin forecast for all test months
pred_test = np.asarray(model_fit.forecast(steps=len(test_data)))

# -------------------------------
# 7. Evaluation Metrics
# -------------------------------
mae = mean_absolute_error(test_data, pred_test)

mape = mean_absolute_percentage_error(test_data, pred_test) * 100

rmse = root_mean_squared_error(test_data, pred_test)

# Seasonal MASE denominator
mae_naive = mean_absolute_error(
    train_data[seasonal_period:], train_data[:-seasonal_period]
)

mase = mae / mae_naive

# Fixed-origin seasonal naive baseline
naive_pred = np.resize(train_data[-seasonal_period:], len(test_data))
mae_baseline = mean_absolute_error(test_data, naive_pred)
mase_naive = mae_baseline / mae_naive

print("\nTesting Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.4f}%")
print(f"RMSE : {rmse:.4f}")
print(f"MASE : {mase:.4f}")

print(f"Seasonal Naive MASE: {mase_naive:.4f}")

# -------------------------------
# 8. Visualization
# -------------------------------
# Monte Carlo simulation
# 从残差中抽样，假设残差服从正态分布，生成未来的预测路径
# 也可以从历史残差中 bootstrap 抽样，生成未来的预测路径
simulated_paths = np.asarray(
    model_fit.simulate(
        nsimulations=len(test_data),
        repetitions=5000,
        error="add",
    )
)

# 95% prediction interval
lower_bound = np.quantile(simulated_paths, 0.025, axis=1)
upper_bound = np.quantile(simulated_paths, 0.975, axis=1)

# Visualization
plt.figure(figsize=(12, 5))

plt.plot(range(len(raw_data)), raw_data, color="blue", label="Actual")

plt.plot(
    range(train_len, len(raw_data)),
    pred_test,
    color="red",
    label="Holt-Winters Forecast",
)

plt.fill_between(
    range(train_len, len(raw_data)),
    lower_bound,
    upper_bound,
    color="red",
    alpha=0.15,
    label="95% Prediction Interval",
)

plt.axvline(x=train_len, linestyle="--", color="gray")

plt.xlabel("Month Index")
plt.ylabel("Passengers")
plt.title("Holt-Winters Forecast with 95% Prediction Interval")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
