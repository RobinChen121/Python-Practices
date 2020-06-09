#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/11 16:36
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description: 读取整理好的 excel数据，并采用不同的预测方法预测

"""


import pandas as pd
import numpy as np
# import xlrd
import os
import grey_model


source = '运维及成本'  # 可以改成其他类别的名字
#source_budgets = [1500000000, 2887110000, 1743225000, 487350000, 950000000]
# direct = os.getcwd() # get the current directory
direct = os.path.dirname(os.getcwd()) # 返回当前目录的父目录
file_direct = os.path.join(direct, 'data-output')
file_name = source + '合并删除增加电商等14-18-金额.xlsx'
file_path_xls = os.path.join(file_direct, file_name)
data_matrix = pd.read_excel(file_path_xls, index_col = 0, encoding = 'gbk')  # gbk for reading Chinese characters
column_num = data_matrix.shape[1]
history_data = data_matrix.iloc[:, 1:column_num - 1].values # not include 2018 data, iloc not inlude the value after :

row_num  = history_data.shape[0]
column_num  = history_data.shape[1]
forecast_2018 = [0 for i in range(row_num)]   # an empty matrix of given size， 记录 2018 年的预测数据
# forcast_rule = ['' for i in range(row_num)]  # 记录预测方法

for i in range(row_num):
    forecast_2018[i] = grey_model.gm(history_data[i, :]) # 选取预测方法


# 应急 1.4，居配取 -0.7，零购 0，信息 0，农网取 0，配网取系数 0.9, 运维乘以 1.5，供电分离系数 4, 大修取 0，基建 0.3,营销 0.25，技改需要取负系数 -0.3
ratio = 1.5   # (source_budgets[-1] - source_budgets[-2]) / source_budgets[-2]
forecast_2018 = np.asarray(forecast_2018) * (1 + ratio)

forecast_final = data_matrix
forecast_final['2018预测值'] = forecast_2018
forecast_final['预测偏差'] = forecast_2018 - data_matrix.iloc[:, column_num + 1].values
gap_rate = (forecast_2018 - data_matrix.iloc[:, column_num + 1].values) / data_matrix.iloc[:, column_num + 1].values
forecast_final['偏差率'] = gap_rate
forecast_final = forecast_final.fillna(0)  # replace nan with 0
forecast_final = forecast_final.replace(np.inf, 1000000)
forecast_final = forecast_final.sort_values(by='偏差率', ascending=False)
filename = source + '2018预测值对比-删除合并增加电商等-金额.xlsx'
forecast_final.to_excel(filename, encoding='gbk')