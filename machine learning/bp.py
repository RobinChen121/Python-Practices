"""
@Python version: 3.12
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 29/11/2025, 19:40
@Desc: vanilla neuro network: batch and adam can increase the accuracy rate.

"""

import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# 1. 加载数据
iris = datasets.load_iris()
X = iris.data  # (150, 4)
y = iris.target  # (150,)

# 2. 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. 转成 tensor
X = torch.tensor(X, dtype=torch.float32)  # [150, 4]
y = torch.tensor(y, dtype=torch.long)  # CrossEntropyLoss 要 long


# 4. 划分训练与测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. 创建 Dataset
class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = IrisDataset(X_train, y_train)
test_dataset = IrisDataset(X_test, y_test)

# 采用 batch 能减少噪声
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)


# 6. 构建分类模型（简单 MLP）
class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=10, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
        )

    # def __init__(self):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Linear(4, 10),
    #         nn.Sigmoid(),
    #         nn.Linear(10, 3),  # 3 类输出
    #     )

    def forward(self, x):
        return self.net(x)


model = IrisNet()

# 7. 损失函数 + 优化器
criterion = nn.CrossEntropyLoss()  # 多分类损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# 8. 训练
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for Xb, yb in train_loader:
        optimizer.zero_grad()
        output = model(Xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 100 == 0:
        print(f"Epoch {epoch+1}, Loss = {total_loss:.4f}")


# 9. 测试集准确率
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for Xb, yb in test_loader:
        output = model(Xb)
        pred = torch.argmax(output, dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)

print(f"\n测试集准确率：{correct / total * 100:.2f}%")
