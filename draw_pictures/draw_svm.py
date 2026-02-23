#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 13:16:03 2026

@author: zhenchen

@Python version: 3.10

@disp:  
    
    
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 生成简单的二维线性可分数据
np.random.seed(42)
X_pos = np.random.randn(10, 2) + [2, 2]   # 正类
X_neg = np.random.randn(10, 2) + [-2, -2] # 负类
X = np.vstack((X_pos, X_neg))
y = np.array([1]*10 + [-1]*10)

# 训练线性SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# 获取超平面参数
w = clf.coef_[0]
b = clf.intercept_[0]
# 计算间隔
margin = 1 / np.linalg.norm(w)

# 绘图
plt.figure(figsize=(8, 6))

# 绘制数据点
plt.scatter(X_pos[:, 0], X_pos[:, 1], color='blue', s=60, label='Positive class')

plt.scatter(X_neg[:, 0], X_neg[:, 1], color='red', s=60, label='Negative class')

# 绘制超平面和间隔
# 决策边界: w·x + b = 0
# 上下边界: w·x + b = ±1
x_plot = np.linspace(-5, 5, 200)
y_decision = -(w[0]*x_plot + b)/w[1]
y_margin_up = -(w[0]*x_plot + b - 1)/w[1]
y_margin_down = -(w[0]*x_plot + b + 1)/w[1]

plt.plot(x_plot, y_decision, 'k--', label='Decision boundary', alpha=0.5)
plt.plot(x_plot, y_margin_up, 'k-', label='Margin')
plt.plot(x_plot, y_margin_down, 'k-')

# 标记支持向量
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='k', label='Support vectors')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
# plt.title('SVM Geometric Illustration')
plt.legend()
# plt.grid(True)
plt.show()
