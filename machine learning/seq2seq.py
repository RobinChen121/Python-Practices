"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/12/2 12:58
Description: 
    

"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 读取 AirPassengers 数据
from statsmodels.datasets import get_rdataset
data = get_rdataset("AirPassengers").data
ts = data['value'].values.astype(float)

# 归一化
scaler = MinMaxScaler()
ts_scaled = scaler.fit_transform(ts.reshape(-1, 1))

# 构建序列
def create_sequences(data, input_len=12, output_len=12):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

input_len = 12
output_len = 12

X, y = create_sequences(ts_scaled, input_len, output_len)

X = torch.tensor(X, dtype=torch.float32)   # [N, 12, 1]
y = torch.tensor(y, dtype=torch.float32)   # [N, 12, 1]

import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        outputs, (h, c) = self.lstm(x)
        return h, c


class Decoder(nn.Module):
    # output_dim=1：每一步的预测是一个值
    def __init__(self, output_len=12, hidden_dim=64, output_dim=1, num_layers=1):
        super().__init__()
        self.output_len = output_len
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h, c):
        # decoder 输入初始化：用 0 向量作为起点
        # h 的形状: [num_layers * num_directions, batch_size, hidden_dim]
        # PyTorch 的 LSTM 输入必须是 3 维张量，格式是：
        # [batch_size, seq_len, input_size]
        decoder_input = torch.zeros((h.size(1), 1, 1))  # [batch, seq=1, feature=1]

        outputs = []
        for _ in range(self.output_len):
            out, (h, c) = self.lstm(decoder_input, (h, c))
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = pred  # 下一个时间步用前一步预测

        outputs = torch.cat(outputs, dim=1)  # -> [batch, 12, 1]
        return outputs


class Seq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(output_len=output_len)

    def forward(self, x):
        h, c = self.encoder(x)
        out = self.decoder(h, c)
        return out

model = Seq2Seq()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs}  Loss = {loss.item():.6f}")

model.eval()

last_input = torch.tensor(ts_scaled, dtype=torch.float32).reshape(1, input_len, 1)

with torch.no_grad():
    pred_scaled = model(last_input).detach().numpy()

prediction = scaler.inverse_transform(pred_scaled[0])

print("Predicted next 12 months:")
print(prediction.flatten())


plt.figure(figsize=(10,5))
plt.plot(ts, label="real")
plt.plot(range(0, len(ts)+12), prediction, marker="o", label="Predicted")
plt.legend()
plt.title("Seq2Seq LSTM Forecast - AirPassengers")
plt.show()
