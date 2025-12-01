"""
@Python version: 3.12
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 30/11/2025, 13:58
@Desc: for the cancer data;

"""

# 导入必要库
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1️⃣ 加载数据
data = load_breast_cancer()
X = data.data
y = data.target

# 2️⃣ 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3️⃣ 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 转为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# 4️⃣ 定义模型
class BreastCancerModel(nn.Module):
    def __init__(self, input_dim):
        super(BreastCancerModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),  # 二分类使用 Sigmoid
        )

    def forward(self, x):
        return self.network(x)


model = BreastCancerModel(X_train.shape[1])  # 因为 input_dim 是初始化的传递参数

# 5️⃣ 定义损失函数和优化器
criterion = nn.BCELoss()  # 二分类
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6️⃣ 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 7️⃣ 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred > 0.5).float()
    acc = accuracy_score(y_test, y_pred_class)
    print(f"Test Accuracy: {acc:.4f}")
