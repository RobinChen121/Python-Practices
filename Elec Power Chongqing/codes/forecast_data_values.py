# -*- coding: utf-8 -*-
"""
Created on Thu May  9 19:36:30 2019

@author: Zhen Chen

@email: okchen321@163.com 

Python version： 3.7

Description: forecast the data based on product costs
            预测数据
    
"""

import pandas as pd
import numpy as np
# import xlrd
import os



source = '运维及成本'
#source_budgets = [1500000000, 2887110000, 1743225000, 487350000, 950000000]
direct = os.getcwd() # get the current directory
file_direct = os.path.join(direct, 'data-output')
file_name = source + '合并删除增加电商等14-18-金额.xlsx'
file_path_xls = os.path.join(file_direct, file_name)
data_matrix = pd.read_excel(file_path_xls, index_col = 0, encoding = 'gbk')  # gbk for reading Chinese characters

column_num = data_matrix.shape[1]
history_data = data_matrix.iloc[:, 1 : column_num - 1].values # not include 2018 data, iloc not inlude the value after :

row_num  = history_data.shape[0]                                  
column_num  = history_data.shape[1]                  
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
    if history_data[i, - 1] < 50000:
        forcast_2018[i] = history_data[i, - 1] / 2
        return
    if arr[column_num - 2] > 8 * arr[column_num - 1] and arr[column_num - 1] > 0.1:
        forcast_2018[i] = arr[column_num - 1] / 5
        forcast_rule[i] = 8
        return
#    if arr[column_num - 1] > 10 * arr[column_num - 2] and arr[column_num - 2] > 0.1:
#        forcast_2018[i] = arr[column_num - 1]
#        forcast_rule[i] = 8
#        return
     # 若上一年小于5万，则按上一年数据处理

    if (max(arr) - min(arr))/ min(arr) > 3: # 剔除极端值
        if (max(arr)) / sum(arr) > 0.5:
            forcast_2018[i] = (sum(arr) - max(arr)) / (len(arr) - 1)
        if (min(arr)) / sum(arr) < 0.1:
            forcast_2018[i] = (sum(arr) - min(arr)) / (len(arr) - 1)
        if (max(arr)) / sum(arr) > 0.5 and (min(arr)) / sum(arr) < 0.1:
            forcast_2018[i] = (sum(arr) - min(arr) - max(arr)) / (len(arr) - 2)
        forcast_rule[i] = 13
    else:
        forcast_2018[i] = arr.mean()
        forcast_rule[i] = 3
    return

def rule4(i, arr):
    forcast_2018[i] = arr[column_num - 1]
    forcast_rule[i] = 4
    return




for i in range(row_num):    
    if  history_data[i, 0] in ['布电线', '钢芯铝绞线、钢绞线', '低压(刀)开关']:
        forcast_2018[i] = history_data[i, column_num - 1] / 3
        continue 
    if  history_data[i, 0] in ['布电线', '10千伏交流避雷器', '10千伏跌落式熔断器', '配变计量箱']:
        forcast_2018[i] = history_data[i, column_num - 1] 
        continue
    # 若连续两年下降10倍，则19年也下降10倍
    if min(history_data[i, column_num - 3 : column_num]) > 1 and history_data[i, column_num - 3] > \
        10 * history_data[i, column_num - 2] and history_data[i, column_num - 2] > 10 * history_data[i, column_num - 1] :
            forcast_2018[i] = history_data[i, column_num - 1] / 10
            forcast_rule[i] = 9
            continue
    # 若 17, 18 年为零，则 19 年也为零                        
    if history_data[i][column_num - 1] < 0.1 and history_data[i][column_num - 2] < 0.1: 
        rule1(i)
        continue                       
    # 如果从14年开始，连续出现两个零以上，之后的数据都不为零，则对不为零的数据移动平均
    for j in range(1, column_num):
        if sum(history_data[i, 0 : j]) < 10 and min(history_data[i, j : column_num]) > 10:
            rule2(i, history_data[i, j : column_num])
            break;
    # 如果出现 5 年均有数据，但波动幅度（最大值-最小值)/最小值 超过3倍
    # 我们按最近 3 年的平均数做预测；如果未超过5倍，则取所有年度的平均值
    # 若预测年度前一年比预测前第二年下降或上涨超过10倍，则取前一年的值
    if min(history_data[i, :]) > 10:
        rule3(i, history_data[i, :])
        continue
    # 如果只有 17 年缺失，18 年出现极端值，则按 18 年的数据处理；
    if history_data[i, column_num - 2] == 0 and min(history_data[i, 0 : column_num - 2]) > 10 \
        and history_data[i, column_num - 1] > 3 * history_data[i, 0 : column_num - 2].mean():
            rule4(i, history_data[i, :])
            continue;
    # 如果只有 14 年为 0，则将 14 年数据剔除，计算规则与 3 相同
    if history_data[i, 0] == 0 and min(history_data[i, 1 : column_num ]) > 10:
        rule3(i, history_data[i, 1 : column_num])
        forcast_rule[i] = 10
        continue
    # 若零交替出现，则按交替规律预测
#    if (sum(history_data[i, range(0, column_num, 2)]) < 10 and min(history_data[i, range(1, column_num, 2)]) > 10) or \
#        (sum(history_data[i, range(1, column_num, 2)]) < 10 and min(history_data[i, range(0, column_num, 2)]) > 10):
#            forcast_2018[i] = history_data[i, column_num - 2]
#            forcast_rule[i] = 6
#            continue
    # 若 18 年不为零，14 年不为零，而其他年存在零，则将零剔除，取剩下值的移动平均
    if history_data[i, column_num - 1] > 10 and history_data[i, 0] > 10\
    and min(history_data[i, 0 : column_num - 1]) < 10:
        arr = history_data[i, :]
        arr = arr[arr > 10]
        forcast_2018[i] = arr.mean()
        forcast_rule[i] = 7
        continue
    
    # 若只有前两年有数据，则取平均值
    if min(history_data[i, column_num - 1], history_data[i, column_num - 2]) > 0.1 and \
       sum(history_data[i, 1 : column_num - 2]) < 0.1:
       arr = history_data[i, :]
       arr = arr[arr > 0.1]
       if arr[-2] > 2.5 * arr[-1]:
           forcast_2018[i] = arr[-1] / 1.5
       else:
           forcast_2018[i] = arr.mean()
       forcast_rule[i] = 12
       continue
    
   
    
    # 其他情况超过两个零按零处理，不适用与供电分离   
    arr = history_data[i, :]
    arr = arr[arr > 0.1]
    if len(arr[arr > 0.1]) >= 2:
        forcast_2018[i] = 0
        forcast_rule[i] = 11

forcast_final = history_data       
ratio = 1.5  #* (source_budgets[-1] - source_budgets[-2]) / source_budgets[-2] # 应急1.4，居配取-0.7，零购0，信息0，农网取0，配网取系数0.9, 运维乘以1.5，供电分离系数4,大修取0，基建0.3,营销0.25，技改需要取负系数-0.3    
forcast_2018 = np.asarray(forcast_2018) * (1 + ratio)     
     
forcast_final['2018预测值'] = forcast_2018
forcast_final['预测偏差'] = forcast_2018 - history_data.iloc[:, column_num + 1].values
gap_rate =  (forcast_2018 - history_data.iloc[:, column_num + 1].values)/history_data.iloc[:, column_num + 1].values
forcast_final['预测规则'] = forcast_rule 
forcast_final['偏差率'] = gap_rate
forcast_final = forcast_final.fillna(0) # replace nan with 0 
forcast_final = forcast_final.replace(np.inf, 1000000)
forcast_final = forcast_final.sort_values(by = '偏差率', ascending = False)
filename = source + '2018预测值对比-删除合并增加电商等-金额.xlsx'
forcast_final.to_excel(filename, encoding = 'gbk')





 

  
     
        

