"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 18/11/2026, 11:51
@Desc: SARIMA forecasting for Weekly CO2 dataset.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)

# -------------------------------
# 1. Configuration
# -------------------------------
train_ratio = 0.8
seasonal_period = 52

order = (1, 1, 1)
seasonal_order = (0, 1, 1, 52)

trend = "n"

# -------------------------------
# 2. Load Weekly CO2
# -------------------------------
df = sm.datasets.co2.load_pandas().data

# Regular weekly observations
# Fill missing values using previous observations
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
# 4. SARIMA Model
# -------------------------------
model = SARIMAX(
    train_data,
    order=order,
    seasonal_order=seasonal_order,
    trend=trend,
    enforce_stationarity=True,
    enforce_invertibility=True,
)

model_fit = model.fit(disp=False, maxiter=300)

print("\nSARIMA Model Summary:")
print(model_fit.summary())

print("\nAIC:", model_fit.aic)
print("BIC:", model_fit.bic)

print("Converged:", model_fit.mle_retvals.get("converged"))

# -------------------------------
# 5. Training Predictions
# -------------------------------
# In-sample one-step-ahead predictions

train_start = model_fit.loglikelihood_burn

train_result = model_fit.get_prediction(
    start=train_start, end=train_len - 1, dynamic=False
)

pred_train = np.asarray(train_result.predicted_mean)

# -------------------------------
# 6. Testing Predictions
# -------------------------------
# Fixed-origin forecasting:
# Do not use actual test observations

forecast_result = model_fit.get_forecast(steps=len(test_data))

pred_test = np.asarray(forecast_result.predicted_mean)

# 95% prediction intervals
conf_int = np.asarray(forecast_result.conf_int(alpha=0.05))

lower_bound = conf_int[:, 0]
upper_bound = conf_int[:, 1]

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

# -------------------------------
# 8. Seasonal Naive Baseline
# -------------------------------
# Fixed-origin seasonal naive:
# Repeat the last 52 training observations

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
# 9. Forecast Results
# -------------------------------
results = pd.DataFrame(
    {
        "Actual": test_data,
        "Predicted": pred_test,
        "Lower 95%": lower_bound,
        "Upper 95%": upper_bound,
    }
)

print("\nForecast Results:")
print(results.round(3).to_string())

# -------------------------------
# 10. Visualization
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

# 95% prediction interval
plt.fill_between(
    range(train_len, len(raw_data)),
    lower_bound,
    upper_bound,
    color="red",
    alpha=0.15,
    label="95% Prediction Interval",
)

# Training/testing split
plt.axvline(x=train_len, color="gray", linestyle="--", label="Train/Test Split")

plt.xlabel("Week Index")
plt.ylabel("CO2 Concentration (ppm)")
plt.title("Weekly CO2: SARIMA Forecast")
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
plt.title("Weekly CO2: SARIMA Testing Forecast")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
