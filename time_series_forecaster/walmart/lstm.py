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

from draw_pic import draw
from hyper_tuning import HyperTuning
from read_data import read_data
from lstm_model import run_lstm

df = read_data()

tune_trial = 20 # number of hyperparameter tuning

# 选择一个 store+dept
store_id = 9
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
file_name = "records/" + file_name
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


tuner = HyperTuning(X, y, scaler, file_name, best_loss)
best_params = tuner.tune(tune_trial)

_, real_pred_train, real_pred_eval = run_lstm(X, y, scaler, file_name, best_loss, **best_params)

draw(sales, seq_len, X, real_pred_train, real_pred_eval)
