"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/12/8 22:53
Description: 
    

"""
from torch.utils.data import DataLoader
from torch import nn
import torch
from sklearn.model_selection import train_test_split
import numpy as np

# define the LSTM model
class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步
        out = self.fc(out)
        return out


def run_lstm(
        X, y, scaler, file_name, best_loss, batch_size, hidden_size, num_layers, bidirectional=False,
        learning_rate=0.001
):
    # --- 在这里进行 split ---
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
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()  # 对于剧烈波动的数据，用 L1Loss 好

    # priority using GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # mac gpu
    else:
        device = torch.device("cpu")
    model.to(device)

    epochs = 30
    real_pred_train = []
    for epoch in range(epochs):
        real_pred_train = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)  # 必须将输入数据和模型放在同一个 device
            batch_y = batch_y.to(device)
            # 这个循环 loader 里面的数据一个一个传进去
            optimizer.zero_grad()
            # lstm 的输入数据维度是 (batch_size, seq_size, input_size)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            # .numpy() 只能用于不需要梯度的 Tensor
            real_pred = scaler.inverse_transform(pred.detach().cpu().numpy())
            real_pred_train.extend(real_pred.flatten())
            loss.backward()
            optimizer.step()

        # print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")

    # model.load_state_dict(torch.load(file_name + "_weights" + ".pth", map_location="cpu"))
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
    if real_loss < best_loss:
        text_name = file_name + "_weights" + ".pth"
        torch.save(model.state_dict(), text_name)
    return real_loss, real_pred_train, real_pred_eval
