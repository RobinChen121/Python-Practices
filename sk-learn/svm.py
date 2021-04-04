# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:17:10 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

cancer = datasets.load_breast_cancer()
cancer_data =  cancer['data']
cancer_target = cancer['target']

cancer_data_train, cancer_data_test, cancer_target_train, \
    cancer_target_test = train_test_split(cancer_data, cancer_target, test_size = 0.2)
    
# 数据标准化
stdScaler = StandardScaler().fit(cancer_data_train)
cancer_trainStd = stdScaler.transform(cancer_data_train)
cancer_testStd = stdScaler.transform(cancer_data_test)

# 建立 SVM 模型
svm = SVC().fit(cancer_trainStd, cancer_target_train)
print('建立的SVM模型为： \n', svm)

# 预测训练集结果
cancer_target_pred = svm.predict(cancer_testStd)
print('预测前20个结果为：\n', cancer_target_pred[:20])

# 预测和真实一样的数目
print('预测对的结果数目为：' , np.sum(cancer_target_pred == cancer_target_test))
print('神经网络预测结果评价报告：\n', classification_report(cancer_target_test,cancer_target_pred))