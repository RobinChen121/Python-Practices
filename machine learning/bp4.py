"""
@Python version: 3.12
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 30/11/2025, 14:01
@Desc: sklearn for iris data;

"""

# 导入库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 2️⃣ 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4️⃣ 定义模型
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=200)

# 5️⃣ 训练模型
model.fit(X_train, y_train)

# 6️⃣ 预测
y_pred = model.predict(X_test)

# 7️⃣ 评估模型
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4%}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
