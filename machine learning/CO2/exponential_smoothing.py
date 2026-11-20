"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 18/11/2026, 11:45
@Desc:

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

# -------------------------------
# 1. Configuration
# -------------------------------
np.random.seed(42)

train_ratio = 0.8
seasonal_period = 52

trend_type = "add"
seasonal_type = "add"

num_simulations = 1000

# -------------------------------
# 2. Load Weekly CO2
# -------------------------------
df = sm.datasets.co2.load_pandas().data

# Regular weekly observations
# Fill missing values using past observations
data = df["co2"].resample("W-SAT").mean().ffill().dropna()

raw_data = data.to_numpy(dtype=float)

print("Dataset length:", len(raw_data))
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

model_fit = model.fit(optimized=True)

print("\nModel Parameters:")
print(model_fit.params)

print("\nModel AIC:", model_fit.aic)
print("Model BIC:", model_fit.bic)
print("Model SSE:", model_fit.sse)

# -------------------------------
# 5. Training Predictions
# -------------------------------
pred_train = np.asarray(model_fit.fittedvalues)

# -------------------------------
# 6. Testing Predictions
# -------------------------------
# Fixed-origin forecasting
# No actual test observations are used

pred_test = np.asarray(model_fit.forecast(steps=len(test_data)))

# -------------------------------
# 7. Prediction Intervals
# -------------------------------
# Monte Carlo simulation
# Shape: (test_length, repetitions)

simulated_paths = np.asarray(
    model_fit.simulate(
        nsimulations=len(test_data),
        repetitions=num_simulations,
        error="add",
        random_errors=None,
    )
)

# 95% prediction intervals
lower_bound = np.quantile(simulated_paths, 0.025, axis=1)

upper_bound = np.quantile(simulated_paths, 0.975, axis=1)

# -------------------------------
# 8. Evaluation Metrics
# -------------------------------
mae = mean_absolute_error(test_data, pred_test)

mape = mean_absolute_percentage_error(test_data, pred_test) * 100

rmse = root_mean_squared_error(test_data, pred_test)

# Seasonal MASE denominator
mae_naive = mean_absolute_error(
    train_data[seasonal_period:], train_data[:-seasonal_period]
)

mase = mae / mae_naive

# -------------------------------
# 9. Seasonal Naive Baseline
# -------------------------------
# Fixed-origin seasonal naive forecast
# Repeat the last 52 training observations
# 用最后一个季节的观测值作为预测值，下面这行代码循环生成与测试集长度相同的预测值
# resize() 会将数组调整为指定长度，如果新长度大于原长度，np.resize() 会循环重复原数组的元素np.
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
# 10. Visualization
# -------------------------------
plt.figure(figsize=(12, 5))

# Actual observations
plt.plot(range(len(raw_data)), raw_data, label="Actual", color="blue")

# Training predictions
plt.plot(range(len(pred_train)), pred_train, label="Predicted (Train)", color="green")

# Testing predictions
plt.plot(
    range(train_len, len(raw_data)), pred_test, label="Predicted (Test)", color="red"
)

# 95% prediction interval
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

plt.xlabel("Week Index")
plt.ylabel("CO2 Concentration (ppm)")
plt.title("Weekly CO2: Holt-Winters Seasonal Forecast")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------
# 11. Test Period Visualization
# -------------------------------
plt.figure(figsize=(12, 5))

test_index = range(train_len, len(raw_data))

plt.plot(test_index, test_data, label="Actual (Test)", color="blue")

plt.plot(test_index, pred_test, label="Predicted (Test)", color="red")

plt.fill_between(
    test_index,
    lower_bound,
    upper_bound,
    color="red",
    alpha=0.15,
    label="95% Prediction Interval",
)

plt.xlabel("Week Index")
plt.ylabel("CO2 Concentration (ppm)")
plt.title("Weekly CO2: Holt-Winters Testing Forecast")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
