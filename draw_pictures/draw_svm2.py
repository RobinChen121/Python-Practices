#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 13:29:38 2026

@author: zhenchen

@Python version: 3.13

@disp:  
    
    
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# 构造简单数据
np.random.seed(42)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel() + 0.1 * np.random.randn(40)  # 带噪声的 sin 曲线

# 创建 SVR 模型
svr = SVR(kernel='rbf', C=100, epsilon=0.1)
svr.fit(X, y)

# 预测
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
y_pred = svr.predict(X_test)

# 绘图
plt.scatter(X, y, color='red', label='Data points')
plt.plot(X_test, y_pred, color='blue', label='SVR prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Support Vector Regression (SVR)')
plt.legend()
plt.show()
