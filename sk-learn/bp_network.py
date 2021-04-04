""" 
# @File  : bp_network.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2020/12/20
# @Desc  : an example for bp ann in predicting，判断用户用水事件是否为洗浴事件

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib # 保存训练模型

## 读取数据
# 训练集与测试集已经在不同的 excel 里面
Xtrain = pd.read_excel('E:/讲课/Python数据分析/《Python数据分析与应用》源数据和代码/Python数据分析与应用/第9章/任务程序/tmp/sj_final.xlsx')
ytrain = pd.read_excel('E:/讲课/Python数据分析/《Python数据分析与应用》源数据和代码/Python数据分析与应用/第9章/任务程序/data/water_heater_log.xlsx')
test = pd.read_excel('E:/讲课/Python数据分析/《Python数据分析与应用》源数据和代码/Python数据分析与应用/第9章/任务程序/data/test_data.xlsx')

## 训练集测试集区分。
x_train, x_test, y_train, y_test = \
Xtrain.iloc[:,5:],test.iloc[:,4:-1],\
ytrain.iloc[:,-1],test.iloc[:,-1]
## 标准化
stdScaler = StandardScaler().fit(x_train) # 生成标准化的规则
x_stdtrain = stdScaler.transform(x_train)
x_stdtest = stdScaler.transform(x_test)

## 建立模型
bpnn = MLPClassifier(hidden_layer_sizes = (20,10),
    max_iter = 200, solver = 'adam',random_state=45)
bpnn.fit(x_stdtrain, y_train)
## 保存模型
joblib.dump(bpnn,'water_heater_nnet.m')
print('构建的模型为：\n',bpnn)

# 模型预测
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#bpnn = joblib.load('water_heater_nnet.m') ## 加载模型
#y_pred = bpnn.predict(x_stdtest) # 返回预测结果
#print('神经网络预测结果评价报告：\n',
#    classification_report(y_test,y_pred))
#
### 绘制roc曲线图
#plt.rcParams['font.sans-serif'] = 'SimHei' ##显示中文
#plt.rcParams['axes.unicode_minus'] = False ##显示负号
#fpr, tpr, thresholds = roc_curve(y_pred,y_test) ## 求出TPR和FPR
#plt.figure(figsize=(6,4))## 创建画布
#plt.plot(fpr,tpr)## 绘制曲线
#plt.title('用户用水事件识别ROC曲线')##标题
#plt.xlabel('FPR')## x轴标签
#plt.ylabel('TPR')## y轴标签
#plt.savefig('用户用水事件识别ROC曲线.png')## 保存图片
#plt.show()## 显示图形


## 测试一下癌症数据
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

# 建立 BP 模型
bpnn = MLPClassifier(hidden_layer_sizes = (20,10),
    max_iter = 200, solver = 'adam',random_state=45)
bpnn.fit(cancer_trainStd, cancer_target_train)

# 预测
y_pred = bpnn.predict(cancer_testStd) # 返回预测结果
print('神经网络预测结果评价报告：\n', classification_report(cancer_target_test,y_pred))


