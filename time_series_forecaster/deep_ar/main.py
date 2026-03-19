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
import torch

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
x_train, y_train, x_test, y_test, v_train, v_test = create_sequence(
    raw_data, embedding_layers, train_length, encoder_length, decoder_length
)

# DataLoader
# 使用 batch size, 若不使用，相当于 full batch
train_dataset = TensorDataset(x_train, y_train, v_train)
test_dataset = TensorDataset(x_test, y_test, v_test)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,  # 训练集通常 shuffle，但时间序列数据除外
    drop_last=False,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# LSTM model
input_size = x_train.shape[-1]
output_size = y_train.shape[-1]
model = DeepARLSTM(
    input_size=input_size,
    output_size=output_size,
    hidden_size=node_num,
    layer_num=layer_num,
)
# set forget bias 初始值 = 1
# 一开始忘掉一半记忆
for name, param in model.named_parameters():
    if "bias" in name:
        n = param.size(0)
        param.data[n // 4 : n // 2].fill_(1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
max_epochs = 1000
model.train()  # 告诉模型处于训练模式
# early stopping criteria
patience = 10
best_loss = float("inf")
counter = 0
for epoch in range(max_epochs):
    epoch_loss = 0.0
    for X_batch, y_batch, v_batch in train_loader:
        optimizer.zero_grad()  # 每次传播时清空梯度，不然会累加
        dist = model(X_batch, v_batch)
        loss = -dist.log_prob(y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)  # 平均每个 batch 的 loss
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        counter = 0
    else:
        counter += 1
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {epoch_loss:.6f}")
    if counter >= patience:
        print(f"Final epoch {epoch + 1}/{max_epochs}, Train Loss: {epoch_loss:.6f}")
        break

# 预测
model.eval()  # 切换到评估模式
with torch.no_grad():
    prediction = model(x_test, v_test)  # 自动调用里面的 forward 函数
    pass
