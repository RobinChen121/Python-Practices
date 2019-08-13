# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:04:09 2019

@author: zhen chen

MIT Licence.

Python version: 3.7

Email: robinchen@swu.edu.cn

Description:  read the quantity data from several excels(new files)
              从新数据文件中读取每一年的销售量
    
"""

import pandas as pd
import numpy as np
import os


source = '农网'

direct = os.getcwd() # get the current directory
file_direct = os.path.join(direct, 'data-files')
file_path_xls = os.path.join(file_direct, "2014年协议库存用量统计.xlsx")
df2014_total = pd.read_excel(file_path_xls, encoding = 'gbk', sheet_name = '使用数量')  # gbk for reading Chinese characters

#print(df.iloc[0:2]) # choose rows, no need to use loc or iloc when choosing columns
df2014 = df2014_total[['行标签', source]]
source_year = source + '2014'
df2014.rename(columns = {'行标签' : '物资品类', source : source_year}, inplace = True)

df2015_total = pd.read_excel(os.path.join(file_direct, "2015年协议库存用量统计.xlsx"), encoding = 'gbk', sheet_name = "使用数量")
df2015 = df2015_total[['行标签', source]]
source_year = source + '2015'
df2015.rename(columns = {'行标签' : '物资品类', source : source_year}, inplace = True)

# merge 14, 15
df_merge_1415 = pd.merge(df2014, df2015, on = '物资品类', how = 'outer')

df2016_total = pd.read_excel(os.path.join(file_direct, "2016年协议库存用量统计.xlsx"), encoding = 'gbk', sheet_name = "使用数量")
df2016 = df2016_total[['行标签', source]]
source_year = source + '2016'
df2016.rename(columns = {'行标签' : '物资品类', source : source_year}, inplace = True)

df_merge_14to16 = pd.merge(df_merge_1415, df2016, on = '物资品类', how = 'outer')
df_merge_14to16 = df_merge_14to16.fillna(0) # replace nan with 0

df2017_total = pd.read_excel(os.path.join(file_direct, "2017年协议库存用量统计.xlsx"), sheet_name = "使用数量", encoding = 'gbk')
df2017 = df2017_total[['行标签', source]]
source_year = source + '2017'
df2017.rename(columns = {'行标签' : '物资品类', source : source_year}, inplace = True)

df_merge_14to17 = pd.merge(df_merge_14to16, df2017, on = '物资品类', how = 'outer')
df_merge_14to17 = df_merge_14to17.fillna(0) # replace nan with 0

df2018_total = pd.read_excel(os.path.join(file_direct, "2018年协议库存用量统计.xlsx"), sheet_name = "使用数量", encoding = 'gbk')
df2018 = df2018_total[['行标签', source]]
source_year = source + '2018'
df2018.rename(columns = {'行标签' : '物资品类', source : source_year}, inplace = True)

df_merge_14to18 = pd.merge(df_merge_14to17, df2018, on = '物资品类', how = 'outer')
df_merge_14to18 = df_merge_14to18.fillna(0) # replace nan with 0 


### to 2017
#filename = source + '初始数据14-17.xlsx'
#df_merge_14to17.to_excel(filename, encoding = 'gbk')

## add other data
direct = os.getcwd() # get the current directory
file_direct = os.path.join(direct, 'data-files')
file_path_xls = os.path.join(file_direct, "14-18年电商数据（金额、数量）.xlsx")
add1 = pd.read_excel(file_path_xls, encoding = 'gbk')
add1 = add1.fillna(0) # replace nan with 0
source_amount = source + '数量'
columns = ['年份', '物资品类', source_amount]
add1_source = add1[columns]
cate_list = add1_source['物资品类'].tolist()
row_num = df_merge_14to18.shape[0]
row_num2 = add1_source.shape[0]
for i in range(row_num):
    if df_merge_14to18['物资品类'][i] in cate_list:
        for j in range(row_num2):
            if add1_source['物资品类'][j] == df_merge_14to18['物资品类'][i]:
                source_year = source + str(add1_source['年份'][j])
                df_merge_14to18[source_year][i] += add1_source[source_amount][j]
                
file_path_xls = os.path.join(file_direct, "14-18年批次数据（金额、数量）.xlsx")
add2 = pd.read_excel(file_path_xls, encoding = 'gbk')
add2 = add2.fillna(0) # replace nan with 0
source_amount = source + '数量'
columns = ['年份', '物资品类', source_amount]
add2_source = add2[columns]
cate_list = add2_source['物资品类'].tolist()
row_num = df_merge_14to18.shape[0]
row_num2 = add2_source.shape[0]
for i in range(row_num):
    if df_merge_14to18['物资品类'][i] in cate_list:
        for j in range(row_num2):
            if add2_source['物资品类'][j] == df_merge_14to18['物资品类'][i]:
                source_year = source + str(add2_source['年份'][j])
                df_merge_14to18[source_year][i] += add2_source[source_amount][j]               
                
file_path_xls = os.path.join(file_direct, "14-18年授权物资（金额、数量）.xlsx")
add3 = pd.read_excel(file_path_xls, encoding = 'gbk')
add3 = add3.fillna(0) # replace nan with 0
source_amount = source + '数量'
columns = ['年份', '物资品类', source_amount]
add3_source = add3[columns]
cate_list = add3_source['物资品类'].tolist()
row_num = df_merge_14to18.shape[0]
row_num2 = add3_source.shape[0]
for i in range(row_num):
    if df_merge_14to18['物资品类'][i] in cate_list:
        for j in range(row_num2):
            if add3_source['物资品类'][j] == df_merge_14to18['物资品类'][i]:
                source_year = source + str(add3_source['年份'][j])
                df_merge_14to18[source_year][i] += add3_source[source_amount][j] 
                
## to 2018
#filename = source + '初始数据14-18.xlsx'
#df_merge_14to18.to_excel(filename, encoding = 'gbk')

# drop the all zero data
columns = [source + '2014', source + '2015', source + '2016', source + '2017', source + '2018']
data_matrix = df_merge_14to18[columns].values
shape = data_matrix.shape
row_num = shape[0]
for i in range(row_num):
    if sum(data_matrix[i, :]) < 0.1:
        df_merge_14to18 = df_merge_14to18.drop([i], axis = 0)


file_path_mergeDelete_xls = os.path.join(file_direct, "物资品类合并.xlsx")
delete_merge = pd.read_excel(file_path_mergeDelete_xls, skipcolumns = [0], encoding = 'gbk')
delete_categories = delete_merge['删除物资品类：'].values.tolist() # for the string need tolist()
for i, row in df_merge_14to18.iterrows():
    if row[0] in delete_categories:
        df_merge_14to18 = df_merge_14to18.drop([i], axis = 0)
    if row[0] == '删除':  # drop a row with category '删除'
        df_merge_14to18 = df_merge_14to18.drop([i], axis = 0)
df_merge_14to18 = df_merge_14to18.reset_index(drop = True) # True if not adding a new index column


# revise some data '普通', '优质', merge some files
df_merge_14to18 = df_merge_14to18.reset_index(drop = True) # True if not adding a new index column
row_num = df_merge_14to18.shape[0]
for i in range(row_num):
    df_merge_14to18['物资品类'][i] = df_merge_14to18['物资品类'][i].strip('-普通')
    df_merge_14to18['物资品类'][i] = df_merge_14to18['物资品类'][i].strip('-优质')
    df_merge_14to18['物资品类'][i] = df_merge_14to18['物资品类'][i].replace('kv', '千伏')
    df_merge_14to18['物资品类'][i] = df_merge_14to18['物资品类'][i].replace('KV', '千伏')
    if df_merge_14to18['物资品类'][i] in ['三跨金具', '安全备份金具']:
        df_merge_14to18['物资品类'][i] = '三跨金具' 
    if df_merge_14to18['物资品类'][i] in ['10千伏油浸式变压器', '10千伏非晶合金变压器']:
        df_merge_14to18['物资品类'][i] = '10千伏油浸式变压器' 
    if df_merge_14to18['物资品类'][i] in ['电缆保护管(CPVC)', '电缆保护管(增强聚丙烯)']:
        df_merge_14to18['物资品类'][i] = '电缆保护管' 
    if df_merge_14to18['物资品类'][i] in ['低压配电箱', '配变低压保护开关箱(平移式)',	'配变低压保护开关箱(熔断式)',	'低压柱上综合配电箱(JP柜)','低压综合配电箱']:
        df_merge_14to18['物资品类'][i] = '低压配电箱' 
    if df_merge_14to18['物资品类'][i] in ['电能表',	'0.2S级三相智能电能表',	'0.5S级三相智能电能表',	'1级三相智能电能表',	'2级单相智能电能表']:
        df_merge_14to18['物资品类'][i] = '电能表'    
    if df_merge_14to18['物资品类'][i] in ['锥形水泥杆', '水泥杆', '超高性能混凝土杆']:
        df_merge_14to18['物资品类'][i] = '水泥杆'
    if df_merge_14to18['物资品类'][i] in ['1千伏架空绝缘导线',	'架空绝缘线']:
        df_merge_14to18['物资品类'][i] = '架空绝缘线' 
    if df_merge_14to18['物资品类'][i] in ['低压电力电缆', '低压电力电缆-配农网']:
        df_merge_14to18['物资品类'][i] = '低压电力电缆'    
    if df_merge_14to18['物资品类'][i] in ['10千伏箱式变电站(美式)','10千伏箱式变电站(欧式)']:
        df_merge_14to18['物资品类'][i] = '10千伏箱式变电站'  
    if df_merge_14to18['物资品类'][i] in ['剩余电流动作保护器(不带箱体)', '剩余电流动作保护器(带箱体)']:
        df_merge_14to18['物资品类'][i] = '剩余电流动作保护器'     
    if df_merge_14to18['物资品类'][i] in ['验电接地环',	'10千伏电力金具']:
        df_merge_14to18['物资品类'][i] = '10千伏电力金具' 
    if df_merge_14to18['物资品类'][i] in ['壁挂式直流电源',	'直流电源(柜式)']:
        df_merge_14to18['物资品类'][i] = '壁挂式直流电源' 
    if df_merge_14to18['物资品类'][i] in ['电缆接线保护盒', '电缆接头保护盒']:
        df_merge_14to18['物资品类'][i] = '电缆接线(接头)保护盒'
    if df_merge_14to18['物资品类'][i] in ['剩余电流保护装置', '剩余电流动作保护器']:
        df_merge_14to18['物资品类'][i] = '剩余电流保护装置'  
    if df_merge_14to18['物资品类'][i] in ['低压开关柜', '低压开关柜(固定分隔式)', '低压开关柜(固定分隔式A类)', '低压开关柜(固定式)']:
        df_merge_14to18['物资品类'][i] = '低压开关柜' 
    

row_num = df_merge_14to18.shape[0]
df_merge_14to18 = df_merge_14to18.groupby('物资品类').sum()
df_merge_14to18 = df_merge_14to18.reset_index()
filename = source + '合并删除增加电商等14-18-数量.xlsx'
df_merge_14to18.to_excel(filename, encoding = 'gbk')