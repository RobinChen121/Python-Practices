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
from deep_ar_class import DeepARLSTM

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
input_size = x_train.shape[2]
output_size = y_train.shape[2]
model = DeepARLSTM(input_size = encoder_length, output_size = decoder_length, hidden_size=node_num,layer_num=layer_num)