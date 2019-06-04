# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:27:42 2019

@author: zhen chen

MIT Licence.

Python version: 3.7

Email: robinchen@swu.edu.cn

Description: to forcast specific source data based on demand quantity
    
"""

import pandas as pd
import numpy as np
import os




source = '农网'
source_budgets = [1500000000, 2887110000, 1743225000, 487350000, 950000000]
direct = os.getcwd() # get the current directory
file_direct = os.path.join(direct, 'data-output')
file_name = source + '合并删除增加电商等14-18-数量.xlsx'
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
    # 前两年变化超过 10 倍，则按前一年数据
    if arr[column_num - 2] > 2.5 * arr[column_num - 1] :
        forcast_2018[i] = arr[column_num - 1] / 1.5
        forcast_rule[i] = 8
        return
#    if arr[column_num - 1] > 10 * arr[column_num - 2] :
#        forcast_2018[i] = arr[column_num - 1]
#        forcast_rule[i] = 8
#        return
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
    # 若连续两年下降10倍，则19年也下降10倍
    if min(data_matrix[i, column_num - 3 : column_num]) > 1 and data_matrix[i, column_num - 3] > \
        10 * data_matrix[i, column_num - 2] and data_matrix[i, column_num - 2] > 10 * data_matrix[i, column_num - 1] :
            forcast_2018[i] = data_matrix[i, column_num - 1] / 10
            forcast_rule[i] = 9
            continue
   
    # 若 18 年为零，则 19 年也为零                        
    if data_matrix[i][column_num - 1] < 0.1 and data_matrix[i][column_num - 2] < 0.1: 
        rule1(i)
        continue                       
    # 如果从14年开始，连续出现两个零以上，之后的数据都不为零，则对不为零的数据移动平均
    for j in range(1, column_num):
        if sum(data_matrix[i, 0 : j]) < 0.1 and min(data_matrix[i, j : column_num]) > 0.1:
            rule2(i, data_matrix[i, j : column_num])
            break;
    # 如果出现 5 年均有数据，但波动幅度（最大值-最小值)/最小值 超过3倍
    # 我们按最近 3 年的平均数做预测；如果未超过5倍，则取所有年度的平均值
    # 若预测年度前一年比预测前第二年下降或上涨超过10倍，则取前一年的值
    if min(data_matrix[i, :]) > 0.1:
        rule3(i, data_matrix[i, :])
        continue
    # 如果只有 17 年缺失，18 年出现极端值，则按 18 年的数据处理；
    if data_matrix[i, column_num - 2] == 0 and min(data_matrix[i, 0 : column_num - 2]) > 0.1 \
        and data_matrix[i, column_num - 1] > 3 * data_matrix[i, 0 : column_num - 2].mean():
            rule4(i, data_matrix[i, :])
            continue;
    # 如果只有 14 年为 0，则将 14 年数据剔除，计算规则与 3 相同
    if data_matrix[i, 0] == 0 and min(data_matrix[i, 1 : column_num ]) > 0.1:
        rule3(i, data_matrix[i, 1 : column_num])
        forcast_rule[i] = 5
        continue
    # 若零交替出现，则按交替规律预测
    if (sum(data_matrix[i, range(0, column_num, 2)]) < 0.1 and min(data_matrix[i, range(1, column_num, 2)]) > 0.1) or \
        (sum(data_matrix[i, range(1, column_num, 2)]) < 0.1 and min(data_matrix[i, range(0, column_num, 2)]) > 0.1):
            forcast_2018[i] = data_matrix[i, column_num - 2]
            forcast_rule[i] = 6
            continue
    # 若 18 年不为零，14 年不为零，而其他年存在零，则将零剔除，取剩下值的移动平均
    if data_matrix[i, column_num - 1] > 0.1 and data_matrix[i, 0] > 0.1\
    and min(data_matrix[i, 0 : column_num - 1]) < 0.1:
        arr = data_matrix[i, :]
        arr = arr[arr > 0.1]
        forcast_2018[i] = arr.mean()
        forcast_rule[i] = 7
        continue
     # 若只有前两年有数据，则取平均值
    if min(data_matrix[i, column_num - 2], data_matrix[i, column_num - 1]) > 1 and \
     sum(data_matrix[i, 0 : column_num - 2]) < 1:
       arr = data_matrix[i, :]
       arr = arr[arr > 1]
       forcast_2018[i] = arr.mean()
       forcast_rule[i] = 9
       continue
   
    # 其他情况按非零均值处理   
    arr = data_matrix[i, :]
    arr = arr[arr > 0.1]
    forcast_2018[i] = arr.mean()
    forcast_rule[i] = 10
    

forcast_final = history_data       
ratio = 0 * (source_budgets[-1] - source_budgets[-2]) / source_budgets[-2] # 取 0.2 负偏差 3千多万   
forcast_2018 = np.asarray(forcast_2018) * (1 + ratio)     
     
forcast_final['2018预测值'] = forcast_2018
forcast_final['预测偏差'] = forcast_2018 - history_data.iloc[:, column_num + 1].values
gap_rate =  (forcast_2018 - history_data.iloc[:, column_num + 1].values)/history_data.iloc[:, column_num + 1].values
forcast_final['预测规则'] = forcast_rule 
forcast_final['偏差率'] = gap_rate
filename = source + '2018预测值对比-删除合并增加电商等-数量.xlsx'
forcast_final.to_excel(filename, encoding = 'gbk')