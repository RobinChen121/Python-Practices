"""
Python version: 3.13
Description:
    Vanilla RNN for AirPassengers forecasting.
    Supports one-step and direct multi-step forecasting.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.datasets import get_rdataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

# -------------------------------
# 1. Configuration
# -------------------------------
torch.manual_seed(42)

epoch_num = 300
seq_length = 12
pred_length = 12  # 1: one-step; 12: multi-step

learning_rate = 0.01
batch_size = 24

hidden_size = 32
num_layers = 1

train_ratio = 0.8


# -------------------------------
# 2. Load Data
# -------------------------------
dataset = get_rdataset("AirPassengers").data
raw_data = dataset["value"].to_numpy()

train_len = int(len(raw_data) * train_ratio)
train_data = raw_data[:train_len]


# -------------------------------
# 3. Normalization
# -------------------------------
# Fit using training observations only
data_min = train_data.min()
data_max = train_data.max()

scaled_data = 2 * (raw_data - data_min) / (data_max - data_min) - 1


def inverse_transform(data):
    return (data + 1) * (data_max - data_min) / 2 + data_min


# -------------------------------
# 4. Create Sequences
# -------------------------------
def create_sequences(data, seq_length, pred_length):
    xs, ys = [], []

    for i in range(len(data) - seq_length - pred_length + 1):
        # X: (seq_length, input_size)
        x = torch.tensor(data[i : i + seq_length], dtype=torch.float32).unsqueeze(-1)

        # y: (pred_length,)
        y = torch.tensor(
            data[i + seq_length : i + seq_length + pred_length], dtype=torch.float32
        )

        xs.append(x)
        ys.append(y)

    return torch.stack(xs), torch.stack(ys)


# Training windows: targets within training period
X_train, y_train = create_sequences(scaled_data[:train_len], seq_length, pred_length)

# Test windows:
# Include last seq_length training observations
X_test, y_test = create_sequences(
    scaled_data[train_len - seq_length :], seq_length, pred_length
)
input_size = X_train.shape[-1]  # 输入特征维度

# print("X_train:", X_train.shape)
# print("y_train:", y_train.shape)
# print("X_test:", X_test.shape)
# print("y_test:", y_test.shape)


# -------------------------------
# 5. DataLoader
# -------------------------------
train_dataset = TensorDataset(X_train, y_train)

test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
)


# -------------------------------
# 6. Define Vanilla RNN
# -------------------------------
class SimpleRNN(nn.Module):

    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):

        if h is None:
            out, h = self.rnn(x)
        else:
            out, h = self.rnn(x, h)

        # h.shape:
        # (num_layers, batch_size, hidden_size)

        # Last hidden state of the last layer
        out = h[-1]

        # Equivalent to:
        # out = out[:, -1, :]

        # Output shape: (batch_size, output_size)
        return x[:, -pred_length:, 0] + self.linear(out)


model = SimpleRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    output_size=pred_length,
)


# -------------------------------
# 7. Loss Function and Optimizer
# -------------------------------
criterion = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# -------------------------------
# 8. Training
# -------------------------------
for epoch in range(epoch_num):

    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        output = model(X_batch)

        # output: (B, pred_length)
        # y_batch: (B, pred_length)
        loss = criterion(output, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * len(X_batch)

    # Average loss over all training samples
    epoch_loss /= len(train_dataset)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epoch_num}, " f"Train Loss: {epoch_loss:.6f}")


# -------------------------------
# 9. Prediction
# -------------------------------
model.eval()

with torch.inference_mode():

    pred_train_norm = model(X_train)
    pred_test_norm = model(X_test)

    pred_train = inverse_transform(pred_train_norm.numpy())

    pred_test = inverse_transform(pred_test_norm.numpy())

    y_train_real = inverse_transform(y_train.numpy())

    y_test_real = inverse_transform(y_test.numpy())


# -------------------------------
# 10. Evaluation
# -------------------------------
rmse_test = root_mean_squared_error(y_test_real.flatten(), pred_test.flatten())

mae_test = mean_absolute_error(y_test_real, pred_test)

mape_test = mean_absolute_percentage_error(y_test_real, pred_test) * 100

# Seasonal MASE
m = 12

mae_naive = np.mean(np.abs(train_data[m:] - train_data[:-m]))

mase_test = mae_test / mae_naive

print(f"\n测试集 RMSE: {rmse_test:.4f}")
print(f"测试集 MAE: {mae_test:.4f}")
print(f"测试集 MAPE: {mape_test:.2f}%")
print(f"测试集 MASE: {mase_test:.4f}")
# compute naive MASE using seasonal naive forecast
naive_pred = inverse_transform(X_test[:, :pred_length, 0].numpy())
mae_baseline = mean_absolute_error(y_test_real, naive_pred)
mase_naive = mae_baseline / mae_naive
print(f"Seasonal Naive MASE: {mase_naive:.4f}")

# -------------------------------
# 11. Visualization
# -------------------------------
plt.figure(figsize=(12, 5))

# Actual observations
plt.plot(range(len(raw_data)), raw_data, label="Actual", color="blue")

# Training predictions: first forecast step
plt.plot(
    range(seq_length, seq_length + len(pred_train)),
    pred_train[:, 0],
    label="Predicted (Train)",
    color="green",
)

# Test predictions: first forecast step
test_pred = pred_test[:, 0]

# Add remaining steps of the final test window
test_pred = np.concatenate([test_pred, pred_test[-1, 1:]])

plt.plot(
    range(train_len, train_len + len(test_pred)),
    test_pred,
    label="Predicted (Test)",
    color="red",
)

plt.axvline(x=train_len, color="gray", linestyle="--", label="Train/Test Split")

plt.xlabel("Month Index")
plt.ylabel("Passengers")
plt.title("AirPassengers Prediction using Vanilla RNN")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
