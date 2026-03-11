"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 02/03/2026, 10:21
@Desc:

"""

from data_cleaning import get_raw_data, create_sequence
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

batch_size = 64
learning_rate = 0.001
layer_num = 3
node_num = 40  # hidden size
encoder_length = 8
decoder_length = 8
train_length = 43
embedding_dim = 32

# create data sequence
raw_data = get_raw_data()
month_num, item_num = raw_data.shape
embedding_layers = nn.Embedding(item_num, embedding_dim)
x_train, y_train, x_test, y_test = create_sequence(
    raw_data, embedding_layers, train_length, encoder_length, decoder_length
)

# DataLoader
# 使用 batch size, 若不使用，相当于 full batch
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,  # 训练集通常 shuffle，但时间序列数据除外
    drop_last=False,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num_, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layer_num_, batch_first=True)
        # 每个时间步 h_t 形状 [num_layers, batch, hidden_size]
        # 用每个时间步最后那个 layer 的h_t 参与下面的映射
        self.linear = nn.Linear(hidden_size, output_size) # 线性输出,输出维度 [batch, output_size]

    def forward(self, x):
        # 返回所有时间步的隐藏状态及最后一个时间步的（h_n, c_m）
        # out 的形状是 (batch_size, seq_size, hidden_size)
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out
