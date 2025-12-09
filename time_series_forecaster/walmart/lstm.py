"""
@Python version: 3.12
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 03/12/2025, 17:18
@Desc:

"""

import numpy as np
import torch
import os

import optuna
from utils.get_pc_info import get_info
from read_data import read_data
from lstm_model import run_lstm

df = read_data()

tune_trial = 20 # number of hyperparameter tuning

# 选择一个 store+dept
store_id = 1
dept_id = 1
ts = df[(df.Store == store_id) & (df.Dept == dept_id)]

sales = ts["Weekly_Sales"].values.astype(np.float32)
is_holiday = ts["IsHoliday"].astype(float).values

# 需要标准化
# 0-1 变量不需要标准化
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
sales_scaled = scaler.fit_transform(sales.reshape(-1, 1))

# file name in recording
file_name = os.path.basename(__file__)
file_name = file_name.split(".py")[0]
file_name += "_dept" + str(dept_id) + "_store" + str(store_id)

# formulate a sequence
seq_len = 12  # 使用过去 12 周预测下一周


def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i: i + seq_len]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# data = np.column_stack((sales_scaled, is_holiday))
X, y = create_sequences(sales_scaled, seq_len)
X = torch.tensor(X)  # (batch, seq, feature)
y = torch.tensor(y)

if os.path.exists(file_name + "_hyperparameter.pth"):
    para_read = torch.load(file_name + "_hyperparameter.pth")
    best_loss = para_read["best_MAE"]
else:
    best_loss = float("inf")


def objective(trial):
    # 超参数采样
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
    num_layers = trial.suggest_int("num_layers", 1, 3)  # sample from 1 to 3
    # dropoutΩpout = trial.suggest_float('dropout', 0.0, 0.5)
    bidirectional = trial.suggest_categorical("bidirectional", [False])
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-4, 1e-2, log=True
    )  # 1e-4 到 1e-2 的范围内用对数采样学习率
    batch_size = trial.suggest_categorical("batch_size", [1])

    loss, _, _ = run_lstm(
        X, y, scaler, file_name, best_loss, batch_size, hidden_size, num_layers, bidirectional, learning_rate,
        for_tuning=False
    )
    return loss


# 创建优化器
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=tune_trial)  # 20次尝试

print("best hyperparameters:", study.best_params)

# 保存 checkpoint
if study.best_value < best_loss:
    best_loss = study.best_value
    torch.save({
        "best_MAE": study.best_value,
        "hidden size": study.best_params["hidden_size"],
        "num_layers": study.best_params["num_layers"],
        "bidirectional": study.best_params["bidirectional"],
        "learning_rate": study.best_params["learning_rate"],
        "batch_size": study.best_params["batch_size"],
    }, file_name + "_hyperparameter" + ".pth")

text_name = file_name + "_pc" + ".txt"
with open(text_name, "a") as f:
    import platform
    import time

    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    f.write(f"date: {formatted_time}\n")
    f.write(f"file name: {file_name}\n")

    info = get_info()
    for k, v in info.items():
        f.write(f"{k}: {v}\n")
    if torch.cuda.is_available():
        f.write(f"GPU: {torch.cuda.is_available()}\n")
    else:
        f.write(f"GPU: {torch.backends.mps.is_available()}\n")
    f.write(f"Python version: {platform.python_version()}\n")
    f.write(f"best hyperparameters:\n{str(study.best_params)}\n")
    f.write(f"real MAE: {str(study.best_value)}\n")
    f.write(f"historical best MAE: {str(best_loss)}\n")
    equals = "==" * 50
    f.write(f"{equals}\n")

_, real_pred_train, real_pred_eval = run_lstm(X, y, scaler, file_name, best_loss, **study.best_params, for_tuning=True)
import matplotlib

matplotlib.use("Qt5Agg")  # 或者 "Qt5Agg"，具体取决于环境中装了哪个: conda install pyqt
import matplotlib.pyplot as plt

plt.plot(sales, label="real sales")
train_size = int(X.shape[0] * 0.8)

plt.plot(
np.arange(train_size + seq_len),
np.concatenate((np.array(sales[:seq_len]), real_pred_train)),
# np.concatenate((np.array(sales[:seq_len]), real_pred.flatten())),
label = "predicted sales train",
)

plt.plot(
    np.arange(len(sales) - len(real_pred_eval), len(sales)),
    np.array(real_pred_eval),
    # np.concatenate((np.array(sales[:seq_len]), real_pred.flatten())),
    label="predicted sales evaluation",
)
plt.legend()
plt.show()
