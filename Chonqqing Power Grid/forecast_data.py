# -*- coding: utf-8 -*-
"""
Created on Thu May  9 19:36:30 2019

@author: Zhen Chen

@email: okchen321@163.com 

Python version： 3.7

Description: forecast the data 
    
"""

import pandas as pd
import numpy as np
# import xlrd
import os



source = '农网'
source_budgets = [1500000000, 2887110000, 1743225000, 487350000, 950000000]
direct = os.getcwd() # get the current directory
file_direct = os.path.join(direct, 'data-output')
file_name = source + '去除优质普通合并数据14-18.xlsx'
file_path_xls = os.path.join(file_direct, file_name)
history_data = pd.read_excel(file_path_xls, index_col = 0, encoding = 'gbk')  # gbk for reading Chinese characters

column_num = history_data.shape[1]
data_matrix = history_data.iloc[:, 1 : column_num - 1].values # not include 2018 data, iloc not inlude the value after :

row_num  = data_matrix.shape[0]                                  
column_num  = data_matrix.shape[1]                  
forcast_2018 = [0 for i in range(row_num)]   # an empty matrix of given size
forcast_rule = ['' for i in range(row_num)]


def rule1(i):
    forcast_2018[i] = 0  
    forcast_rule[i] = 1
    return 
    
def rule2(i, arr):
    forcast_2018[i] = arr.mean()
    forcast_rule[i] = 2
    return

def rule3(i, arr):
    column_num = len(arr)
    if (max(arr) - min(arr))/ min(arr) > 3:
        forcast_2018[i] = arr[column_num - 3 : column_num].mean() # 平均 2 年还是 3 年
    else:
        forcast_2018[i] = arr.mean()
    forcast_rule[i] = 3
    return

def rule4(i, arr):
    forcast_2018[i] = arr[column_num - 1]
    forcast_rule[i] = 4
    return

for i in range(row_num):    
    # 若 18 年为零，则 19 年也为零                        
    if data_matrix[i][column_num - 1] == 0: 
        rule1(i)
        continue                       
    # 如果从14年开始，连续出现两个零以上，之后的数据都不为零，则对不为零的数据移动平均
    for j in range(1, column_num):
        if sum(data_matrix[i, 0 : j]) < 10 and min(data_matrix[i, j : column_num]) > 10:
            rule2(i, data_matrix[i, j : column_num])
            break;
    # 如果出现 5 年均有数据，但波动幅度（最大值-最小值)/最小值 超过3倍
    # 我们按最近 3 年的平均数做预测；如果未超过5倍，则取所有年度的平均值
    if min(data_matrix[i, :]) > 10:
        rule3(i, data_matrix[i, :])
        continue
    # 如果只有 17 年缺失，18 年出现极端值，则按 18 年的数据处理；
    if data_matrix[i, column_num - 2] == 0 and min(data_matrix[i, 0 : column_num - 2]) > 10 \
        and data_matrix[i, column_num - 1] > 3 * data_matrix[i, 0 : column_num - 2].mean():
            rule4(i, data_matrix[i, :])
            continue;
    # 如果只有 14 年为 0，则将 14 年数据剔除，计算规则与 3 相同
    if data_matrix[i, 0] == 0 and min(data_matrix[i, 1 : column_num ]) > 10:
        rule3(i, data_matrix[i, 1 : column_num])
        forcast_rule[i] = 5
        continue
    # 若零交替出现，则按交替规律预测
    if (sum(data_matrix[i, range(0, column_num, 2)]) < 10 and min(data_matrix[i, range(1, column_num, 2)]) > 10) or \
        (sum(data_matrix[i, range(1, column_num, 2)]) < 10 and min(data_matrix[i, range(0, column_num, 2)]) > 10):
            forcast_2018[i] = data_matrix[i, column_num - 2]
            forcast_rule[i] = 6
            continue
    # 若 18 年不为零，14 年不为零，而其他年存在零，则将零剔除，取剩下值的移动平均
    if data_matrix[i, column_num - 1] > 10 and data_matrix[i, 0] > 10\
    and min(data_matrix[i, 0 : column_num - 1]) < 10:
        arr = data_matrix[i, :]
        arr = arr[arr > 10]
        forcast_2018[i] = arr.mean()
        forcast_rule[i] = 7
        continue
        

forcast_final = history_data       
ratio = 0.2 * (source_budgets[-1] - source_budgets[-2]) / source_budgets[-2] # 取 0.2 负偏差 3千多万   
forcast_2018 = np.asarray(forcast_2018) * (1 + ratio)           
forcast_final['2018预测值'] = forcast_2018
forcast_final['预测偏差'] = forcast_2018 - history_data.iloc[:, column_num + 1].values
forcast_final['预测规则'] = forcast_rule 
filename = source + '2018预测值对比.xlsx'
forcast_final.to_excel(filename, encoding = 'gbk')





 

  
     
        

