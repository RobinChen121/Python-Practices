"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/12/11 13:56
Description: this is to run the network weights
    

"""
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from read_data import read_data
import torch
import os
import numpy as np
from lstm_model import LSTMModel

df = read_data()

# 选择一个 store+dept
store_id = 5
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
file_name = "records/" + "lstm"
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

# priority using GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # mac gpu
else:
    device = torch.device("cpu")


if os.path.exists(file_name + "_hyperparameter.pth"):
    para_read = torch.load(file_name + "_hyperparameter.pth")
    best_loss = para_read["best_MAE"]
    batch_size = para_read["batch_size"]
    hidden_size = para_read["hidden_size"]
    num_layers = para_read["num_layers"]
    bidirectional = para_read["bidirectional"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, shuffle=False
    )

    # 保存为类属性（可选）
    train_set = torch.utils.data.TensorDataset(X_train, y_train)
    eval_set = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_size=X.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
    )
    model.to(device)
    criterion = nn.L1Loss()  # 对于剧烈波动的数据，用 L1Loss 好

    model.load_state_dict(torch.load(file_name + "_weights" + ".pth"))
    model.eval()
    real_pred_eval = []
    real_loss = 0.0
    norm_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch_x, batch_y in eval_loader:
            batch_x = batch_x.to(device)  # 必须将输入数据和模型放在同一个 device
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            norm_loss += loss.item() * batch_size

            real_y = scaler.inverse_transform(batch_y.cpu().numpy())
            real_pred = scaler.inverse_transform(pred.cpu().numpy())

            batch_real_mae = np.abs(real_pred - real_y).mean()
            real_loss += batch_real_mae * batch_size

            real_pred_eval.extend(real_pred.flatten())
            total_samples += batch_size
    norm_loss /= total_samples
    print(f"MAE: {norm_loss:.4f}")
    real_loss /= total_samples
    print(f"real MAE: {real_loss:.4f}")

    import matplotlib

    matplotlib.use("Qt5Agg")  # 或者 "Qt5Agg"，具体取决于环境中装了哪个: conda install pyqt
    import matplotlib.pyplot as plt

    plt.plot(sales, label="real sales")
    train_size = int(X.shape[0] * 0.8)

    plt.plot(
        np.arange(len(sales) - len(real_pred_eval), len(sales)),
        np.array(real_pred_eval),
        # np.concatenate((np.array(sales[:seq_len]), real_pred.flatten())),
        label="predicted sales evaluation",
    )
    plt.legend()
    plt.show()
