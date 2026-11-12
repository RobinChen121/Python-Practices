"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 10/11/2026, 12:40
@Desc: the bp network including validation

"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ==================================================
# Step 1: Raw Data
# ==================================================

iris = load_iris()

X = iris.data
y = iris.target

print("Features:", X.shape)
print("Labels:", y.shape)
print("Classes:", iris.target_names)


# ==================================================
# Step 2: Data Preprocessing
# ==================================================

# Train (60%), Validation (20%), Test (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y,  # stratify 参数用于按类别分层抽样，让拆分后的训练集和测试集尽量保持原数据中各类别的比例
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Standardization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ==================================================
# Step 3: Convert to PyTorch Tensors
# ==================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# ==================================================
# Step 4: Dataset
# ==================================================

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)


# ==================================================
# Step 5: DataLoader
# ==================================================

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ==================================================
# Model Setup: Define the Neural Network
# ==================================================


class IrisNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)


model = IrisNN()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# ==================================================
# Steps 6 & 7: Training & Validation Loop
# ==================================================

num_epochs = 100

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_model_state = None

for epoch in range(num_epochs):

    # ----------------------------------------------
    # Step 6: Training Phase
    # ----------------------------------------------

    model.train()

    running_train_loss = 0.0

    for batch_x, batch_y in train_loader:

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(batch_x)

        # Compute training loss
        loss = criterion(predictions, batch_y)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # .item() 的核心作用：把单元素张量转换成普通 Python 数字
        running_train_loss += loss.item() * batch_x.size(0)

    train_loss = running_train_loss / len(train_dataset)
    train_losses.append(train_loss)

    # ----------------------------------------------
    # Step 7: Validation Phase
    # ----------------------------------------------

    model.eval()  # 切换到评估模式

    running_val_loss = 0.0

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            predictions = model(batch_x)

            loss = criterion(predictions, batch_y)

            running_val_loss += loss.item() * batch_x.size(0)

    val_loss = running_val_loss / len(val_dataset)
    val_losses.append(val_loss)

    # Model selection based on validation loss
    # select the best model parameters among all epochs based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = {
            # detach().clone() 的作用是创建一个新的张量，该张量与原始张量共享相同的数据，但不会计算梯度。这样可以避免在保存模型状态时意外地保留梯度信息，从而节省内存并提高效率。
            # state_dict() 返回一个包含模型所有参数的字典，包括权重，偏置等，其中键是参数的名称，值是对应的张量。通过使用 detach().clone()，我们确保保存的是参数的当前值，而不是与计算图相关联的张量。
            k: v.detach().clone()
            for k, v in model.state_dict().items()
        }

    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch+1:3d}/{num_epochs}, "
            f"Training Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}"
        )


# Restore the best model
model.load_state_dict(best_model_state)


# ==================================================
# Step 8: Model Testing
# ==================================================

model.eval()

all_predictions = []
all_targets = []

with torch.no_grad():

    for batch_x, batch_y in test_loader:

        outputs = model(batch_x)

        predicted_classes = torch.argmax(outputs, dim=1)

        all_predictions.extend(predicted_classes.cpu().numpy())

        all_targets.extend(batch_y.cpu().numpy())

test_accuracy = accuracy_score(all_targets, all_predictions)

print(f"\nTest Accuracy: {test_accuracy:.4f}")

print(
    classification_report(all_targets, all_predictions, target_names=iris.target_names)
)


# ==================================================
# Step 9: Prediction / Inference
# ==================================================

# One new flower:
# Sepal length, Sepal width,
# Petal length, Petal width

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

# Apply the same preprocessing
new_flower = scaler.transform(new_flower)

# Convert to Tensor
new_flower = torch.tensor(new_flower, dtype=torch.float32)

model.eval()
# 为什么不用 torch.no_grad()？因为 torch.inference_mode() 是在 PyTorch 1.9
# 中引入的一个新的上下文管理器，它是专门为推理（inference）设计的。
# 与 torch.no_grad() 类似，它也会禁用梯度计算，从而节省内存和计算资源。
# 然而，torch.inference_mode() 还会进一步优化模型的执行，
# 尤其是在涉及到某些操作（如 BatchNorm 和 Dropout）时，
# 它会自动切换到推理模式，从而提高推理速度和效率。因此，在进行模型推理时，
# 使用 torch.inference_mode() 是更推荐的做法。
with torch.inference_mode():
    logits = model(new_flower)

    probabilities = torch.softmax(
        logits, dim=1
    )  # dim=1 表示按行计算 softmax，dim=0 表示按列计算 softmax

    # 如果不需要输出概率的话，则不需要使用softmax，直接用argmax即可
    predicted_class = torch.argmax(probabilities, dim=1).item()

print("\nPredicted Species:", iris.target_names[predicted_class])

print("Class Probabilities:", probabilities.numpy())


# ==================================================
# Visualize Training and Validation Loss
# ==================================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(train_losses, label="Training Loss")

plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
