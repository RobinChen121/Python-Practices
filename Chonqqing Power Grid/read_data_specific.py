# -*- coding: utf-8 -*-
"""
Created on Thu May  9 19:47:57 2019

@author: Zhen Chen

@email: okchen321@163.com 

Python version： 3.7

Description: read data for specific category in the excels
    
"""

import pandas as pd
import numpy as np
# import xlrd
import os

source = '农网'

direct = os.getcwd() # get the current directory
file_direct = os.path.join(direct, 'data-files')
file_path_xls = os.path.join(file_direct, "2014年协议库存使用量.xlsx")
file_path_csv = os.path.join(file_direct, "2014年协议库存使用量.csv")
df2014_total = pd.read_excel(file_path_xls, encoding = 'gbk')  # gbk for reading Chinese characters

#print(df.iloc[0:2]) # choose rows, no need to use loc or iloc when choosing columns
df2014_total = df2014_total.drop([0], axis = 0) # need df =  in a functio when using drop
df2014_total = df2014_total.drop(columns = ['行标签'], axis = 1)
df2014 = df2014_total[['行标签.1', source]]
source_year = source + '2014'
df2014.rename(columns = {'行标签.1' : '物资品类', source : source_year}, inplace = True)

df2015_total = pd.read_excel(os.path.join(file_direct, "2015年协议库存使用量.xlsx"), skiprows = [0], encoding = 'gbk')
df2015_total = df2015_total.drop(columns = ['行标签'], axis = 1)
df2015_total = df2015_total.drop([0], axis = 0)
df2015 = df2015_total[['行标签.1', source]]
source_year = source + '2015'
df2015.rename(columns = {'行标签.1' : '物资品类', source : source_year}, inplace = True)

# merge 14, 15
df_merge_1415 = pd.merge(df2014, df2015, on = '物资品类', how = 'outer')

df2016_total = pd.read_excel(os.path.join(file_direct, "2016年协议库存使用量.xlsx"), skiprows = [0], encoding = 'gbk')
df2016_total = df2016_total.drop([179, 180], axis = 0)
df2016 = df2016_total[['行标签', '农网']]
source_year = source + '2016'
df2016.rename(columns = {'行标签' : '物资品类', source : source_year}, inplace = True)

df_merge_14to16 = pd.merge(df_merge_1415, df2016, on = '物资品类', how = 'outer')
df_merge_14to16 = df_merge_14to16.fillna({'2014': 0, '2015': 0, '2016': 0}) # replace nan with 0
df_merge_14to16 = df_merge_14to16.sort_values(by = ['物资品类'])
df_merge_14to16 = df_merge_14to16.reset_index(drop = True)  #  use the drop parameter to avoid the old index being added as a column

df2017_total = pd.read_excel(os.path.join(file_direct, "2017年协议库存使用量.xlsx"), encoding = 'gbk')
df2017 = df2017_total[['行标签', '农网']]
source_year = source + '2017'
df2017.rename(columns = {'行标签' : '物资品类', source : source_year}, inplace = True)

df_merge_14to17 = pd.merge(df_merge_14to16, df2017, on = '物资品类', how = 'outer')
df_merge_14to17 = df_merge_14to17.fillna(0) # replace nan with 0
df_merge_14to17 = df_merge_14to17.sort_values(by = ['物资品类'])
df_merge_14to17 = df_merge_14to17.reset_index(drop = True)  

df2018_total = pd.read_excel(os.path.join(file_direct, "2018年协议库存使用量.xlsx"), encoding = 'gbk')
df2018 = df2018_total[['行标签', '农网']]
source_year = source + '2018'
df2018.rename(columns = {'行标签' : '物资品类', source : source_year}, inplace = True)


df_merge_14to18 = pd.merge(df_merge_14to17, df2018, on = '物资品类', how = 'outer')
df_merge_14to18 = df_merge_14to18.fillna(0) # replace nan with 0
df_merge_14to18 = df_merge_14to18.sort_values(by = ['物资品类'])
df_merge_14to18 = df_merge_14to18.reset_index(drop = True)  


## to 2017
filename = source + '初始数据14-17.xlsx'
df_merge_14to17.to_excel(filename, encoding = 'gbk')

# drop the all zero data
columns = [source + '2014', source + '2015', source + '2016', source + '2017']
data_matrix = df_merge_14to17[columns].values
shape = data_matrix.shape
row_num = shape[0]
for i in range(row_num):
    if sum(data_matrix[i, :]) < 10:
        df_merge_14to17 = df_merge_14to17.drop([i], axis = 0)

    
# revise some data '普通', '优质'
df_merge_14to17 = df_merge_14to17.reset_index(drop = True) # True if not adding a new index column
row_num = df_merge_14to17.shape[0]
for i in range(row_num):
    df_merge_14to17['物资品类'][i] = df_merge_14to17['物资品类'][i].strip('-普通')
    df_merge_14to17['物资品类'][i] = df_merge_14to17['物资品类'][i].strip('-优质')
    if df_merge_14to17['物资品类'][i] == '删除':  # drop a row with category '删除'
        df_merge_14to17 = df_merge_14to17.drop([i], axis = 0)
    
df_merge_14to17 = df_merge_14to17.reset_index(drop = True) # True if not adding a new index column
row_num = df_merge_14to17.shape[0]
for i in range(row_num):
    df_merge_14to17['物资品类'][i] = df_merge_14to17['物资品类'][i].replace('kv', '千伏') 
    df_merge_14to17['物资品类'][i] = df_merge_14to17['物资品类'][i].replace('KV', '千伏')  
df_merge_14to17 = df_merge_14to17.groupby('物资品类').sum()
df_merge_14to17 = df_merge_14to17.reset_index()
filename = source + '去除优质普通合并数据14-17.xlsx'
df_merge_14to17.to_excel(filename, encoding = 'gbk')


# to 2018
filename = source + '初始数据14-18.xlsx'
df_merge_14to18.to_excel(filename, encoding = 'gbk')

# drop the all zero data
columns = [source + '2014', source + '2015', source + '2016', source + '2017', source + '2018']
data_matrix = df_merge_14to18[columns].values
shape = data_matrix.shape
row_num = shape[0]
for i in range(row_num):
    if sum(data_matrix[i, :]) < 10:
        df_merge_14to18 = df_merge_14to18.drop([i], axis = 0)

# revise some data '普通', '优质'
df_merge_14to18 = df_merge_14to18.reset_index(drop = True) # True if not adding a new index column
row_num = df_merge_14to18.shape[0]
for i in range(row_num):
    df_merge_14to18['物资品类'][i] = df_merge_14to18['物资品类'][i].strip('-普通')
    df_merge_14to18['物资品类'][i] = df_merge_14to18['物资品类'][i].strip('-优质')
    if df_merge_14to18['物资品类'][i] == '删除':  # drop a row with category '删除'
        df_merge_14to18 = df_merge_14to18.drop([i], axis = 0)
    
df_merge_14to18 = df_merge_14to18.reset_index(drop = True) # True if not adding a new index column
row_num = df_merge_14to18.shape[0]
for i in range(row_num):
    df_merge_14to18['物资品类'][i] = df_merge_14to18['物资品类'][i].replace('kv', '千伏')
    df_merge_14to18['物资品类'][i] = df_merge_14to18['物资品类'][i].replace('KV', '千伏')
df_merge_14to18 = df_merge_14to18.groupby('物资品类').sum()
df_merge_14to18 = df_merge_14to18.reset_index()
filename = source + '去除优质普通合并数据14-18.xlsx'
df_merge_14to18.to_excel(filename, encoding = 'gbk')