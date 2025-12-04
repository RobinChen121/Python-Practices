"""
@Python version: 3.12
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 03/12/2025, 17:18
@Desc:

"""

import numpy as np
from read_data import read_data
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

df = read_data()

# 选择一个 store+dept
store_id = 1
dept_id = 1
ts = df[(df.Store == store_id) & (df.Dept == dept_id)]

sales = ts["Weekly_Sales"].values.astype(np.float32)

# 需要标准化
from sklearn.preprocessing import MinMaxScaler

sales_scaled = MinMaxScaler().fit_transform(sales.reshape(-1, 1))

# formulate a sequence
seq_len = 12  # 使用过去 12 周预测下一周


def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i : i + seq_len]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


X, y = create_sequences(sales_scaled, seq_len)
X = torch.tensor(X)  # (batch, seq, feature)
y = torch.tensor(y)

dataset = torch.utils.data.TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=1, shuffle=False)


# define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out


model = LSTMModel()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


epochs = 30
for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        # lstm 的输入数据维度是 (batch_size, seq_size, input_size)
        pred = model(batch_x)
        loss = criterion(pred.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    pred = model(X)
    loss = criterion(pred.squeeze(), y)
print(f"MAE: {loss.item():.4f}")

import matplotlib.pyplot as plt

plt.plot(sales)
plt.plot(pred.squeeze().detach().numpy())
plt.show()
