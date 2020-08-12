# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:13:52 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

cancer = datasets.load_breast_cancer()
cancer_data = cancer['data']
caner_target = cancer['target']

# 默认测试集大小 0.25
cancer_train, cancer_test, y_train, y_test = train_test_split(cancer_data, cancer_data, test_size = 0.3, random_state = 100)

Scaler = MinMaxScaler().fit(cancer_train) # 生成规则
# 将规则应用于训练集
cancer_train_scale = Scaler.transform(cancer_train)
# 将规则应用于测试集
cancer_test_scale = Scaler.transform(cancer_test)

# 输出测试集预处理后的最大值最小值信息
print(cancer_test_scale.min(axis = 0))
print(cancer_test_scale.max(axis = 0))