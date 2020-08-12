import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = 'SimHei' ## 设置中文显示
plt.rcParams['axes.unicode_minus'] = False
data = np.load(\
               'E:\讲课\Python数据分析\《Python数据分析与应用》源数据和代码\Python数据分析与应用\第3章\任务程序\data\国民经济核算季度数据.npz',\
               allow_pickle=True)
name = data['columns']## 提取其中的columns数组，视为数据的标签
values = data['values']## 提取其中的values数组，数据的存在位置
temp = values.tolist
p = plt.figure(figsize=(12,12)) ##设置画布

import pandas as pd
df= pd.DataFrame.from_dict({item: data[item] for item in data.files}, orient='index')

## 子图2
ax2 = p.add_subplot(2,1,2)
plt.scatter(values[:,0],values[:,6], marker='o',c='r')## 绘制散点
plt.scatter(values[:,0],values[:,7], marker='D',c='b')## 绘制散点
plt.scatter(values[:,0],values[:,8], marker='v',c='y')## 绘制散点
plt.scatter(values[:,0],values[:,9], marker='8',c='g')## 绘制散点
plt.scatter(values[:,0],values[:,10], marker='p',c='c')## 绘制散点
plt.scatter(values[:,0],values[:,11], marker='+',c='m')## 绘制散点
plt.scatter(values[:,0],values[:,12], marker='s',c='k')## 绘制散点
## 绘制散点
plt.scatter(values[:,0],values[:,13], marker='*',c='purple')
## 绘制散点
plt.scatter(values[:,0],values[:,14], marker='d',c='brown')
plt.legend(['农业','工业','建筑','批发','交通',
        '餐饮','金融','房地产','其他']) 
plt.xlabel('年份')## 添加横轴标签
plt.ylabel('生产总值（亿元）')## 添加纵轴标签
plt.xticks(range(0,70,4),values[range(0,70,4),1],rotation=45)
plt.show()