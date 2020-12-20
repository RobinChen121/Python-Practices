# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:35:10 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

from sklearn.linear_model import LogisticRegression
import numpy as np

x = np.array([0.5,0.75,1,1.25,1.5,1.75,1.75,2,.25,2.5,2.75,3,3.25,3.5,4,4.25,4.5,4.75,5,5.5])
x = x.reshape(-1, 1) # 需要转化成列数组
y = [0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1]

model = LogisticRegression().fit(x, y)
print(model.coef_) # 打印出截距
print(model.intercept_) # 打印出斜率

print(model.predict(np.array([2]).reshape(-1,1))) # 预测 x 为 2 时的通过情况