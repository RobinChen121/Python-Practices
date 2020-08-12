# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:32:08 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

from sklearn.model_selection import train_test_split
from sklearn import datasets

cancer = datasets.load_breast_cancer()
cancer_data = cancer['data']
caner_target = cancer['target']
# 默认测试集大小 0.25
x_train, x_test, y_train, y_test = train_test_split(cancer_data, cancer_data, test_size = 0.3, random_state = 100)



