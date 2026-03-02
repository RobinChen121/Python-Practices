#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 22:26:18 2026

@author: zhenchen

@Python version: 3.13

@disp:  
    
    
"""

# ==============================
# Iris 数据集随机森林分类完整示例
# ==============================

# 1️⃣ 导入库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------------
# 2️⃣ 加载数据
# ------------------------------
iris = load_iris()
X = iris.data  # 特征：4 个
y = iris.target  # 类别：0,1,2

# 可选：放入 DataFrame 查看
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print("数据预览：")
print(df.head())

# ------------------------------
# 3️⃣ 划分训练集/测试集
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ------------------------------
# 4️⃣ 训练随机森林
# ------------------------------
rf = RandomForestClassifier(
    n_estimators=100,      # 森林中树的数量
    random_state=42
)
rf.fit(X_train, y_train)

# ------------------------------
# 5️⃣ 预测
# ------------------------------
y_pred = rf.predict(X_test)

# ------------------------------
# 6️⃣ 显示整体准确率
# ------------------------------
acc = accuracy_score(y_test, y_pred)
print(f"\n随机森林在测试集上的准确率: {acc:.2f}")

# ------------------------------
# 7️⃣ 混淆矩阵可视化
# ------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ------------------------------
# 8️⃣ 分类详细指标
# ------------------------------
print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ------------------------------
# 9️⃣ 特征重要性
# ------------------------------
feat_importance = pd.Series(rf.feature_importances_, index=iris.feature_names)
feat_importance.sort_values(ascending=False).plot(kind='bar')
plt.title("Feature Importance")
plt.ylabel("Importance")
plt.show()
