# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 22:37:21 2019

@author: zhen chen

Python version: 3.7

Description: read data from the excel files

xlrd package read very slow for large xls files

pandas can read excel files directly
    
"""

import pandas as pd
import numpy as np
# import xlrd
import os

direct = os.getcwd() # get the current directory
file_direct = os.path.join(direct, 'data-files')
file_path_xls = os.path.join(file_direct, "2014年协议库存使用量.xlsx")
file_path_csv = os.path.join(file_direct, "2014年协议库存使用量.csv")
df2014_total = pd.read_excel(file_path_xls, encoding = 'gbk')  # gbk for reading Chinese characters

#print(df.iloc[0:2]) # choose rows, no need to use loc or iloc when choosing columns
df2014_total = df2014_total.drop([0], axis = 0) # need df =  in a functio when using drop
df2014_total = df2014_total.drop(columns = ['行标签'], axis = 1)
df2014 = df2014_total[['行标签.1', '单项汇总']]
df2014.rename(columns = {'行标签.1' : '物资品类', '单项汇总' : '2014'}, inplace = True)

df2015_total = pd.read_excel(os.path.join(file_direct, "2015年协议库存使用量.xlsx"), skiprows = [0], encoding = 'gbk')
df2015_total = df2015_total.drop(columns = ['行标签'], axis = 1)
df2015_total = df2015_total.drop([0], axis = 0)
df2015 = df2015_total[['行标签.1', '单项汇总']]
df2015.rename(columns = {'行标签.1' : '物资品类', '单项汇总' : '2015'}, inplace = True)

# merge 14, 15
df_merge_1415 = pd.merge(df2014, df2015, on = '物资品类', how = 'outer')

df2016_total = pd.read_excel(os.path.join(file_direct, "2016年协议库存使用量.xlsx"), skiprows = [0], encoding = 'gbk')
df2016_total = df2016_total.drop([179, 180], axis = 0)
df2016 = df2016_total[['行标签', '总计']]
df2016.rename(columns = {'行标签' : '物资品类', '总计' : '2016'}, inplace = True)

df_merge_14to16 = pd.merge(df_merge_1415, df2016, on = '物资品类', how = 'outer')
df_merge_14to16 = df_merge_14to16.fillna({'2014': 0, '2015': 0, '2016': 0}) # replace nan with 0
df_merge_14to16 = df_merge_14to16.sort_values(by = ['物资品类'])
df_merge_14to16 = df_merge_14to16.reset_index(drop = True)  #  use the drop parameter to avoid the old index being added as a column

df2017_total = pd.read_excel(os.path.join(file_direct, "2017年协议库存使用量.xlsx"), encoding = 'gbk')
df2017 = df2017_total[['行标签', '总计']]
df2017.rename(columns = {'行标签' : '物资品类', '总计' : '2017'}, inplace = True)

df_merge_14to17 = pd.merge(df_merge_14to16, df2017, on = '物资品类', how = 'outer')
df_merge_14to17 = df_merge_14to17.fillna(0) # replace nan with 0
df_merge_14to17 = df_merge_14to17.sort_values(by = ['物资品类'])
df_merge_14to17 = df_merge_14to17.reset_index(drop = True)  


df2018_total = pd.read_excel(os.path.join(file_direct, "2018年协议库存使用量.xlsx"), encoding = 'gbk')
df2018 = df2018_total[['行标签', '总计']]
df2018.rename(columns = {'行标签' : '物资品类', '总计' : '2018'}, inplace = True)


df_merge_14to18 = pd.merge(df_merge_14to17, df2018, on = '物资品类', how = 'outer')
df_merge_14to18 = df_merge_14to18.fillna(0) # replace nan with 0
df_merge_14to18 = df_merge_14to18.sort_values(by = ['物资品类'])
df_merge_14to18 = df_merge_14to18.reset_index(drop = True)  

df_merge_14to18['变异系数'] = 0  # add a new column


data_matrix = df_merge_14to18.as_matrix(columns = ['2014', '2015', '2016', '2017', '2018'])
shape = data_matrix.shape
row_num = shape[0]
column_num = shape[1]

vc = np.empty((row_num, 1))   # an empty matrix of given size

for i in range(row_num) :   #  compute variable coefficients
    vc[i] = np.std(data_matrix[i, :]) / np.mean(data_matrix[i, :])

df_merge_14to18['变异系数'] = vc

df_merge_14to18.to_csv('merge_14to18.csv', encoding = 'gbk')
df_merge_14to18.to_excel('整合数据14-18.xlsx', encoding = 'gbk')



## open the workbook
#book = xlrd.open_workbook(file_path_xls)
#
## get the worksheet by index
#sheet = book.sheet_by_index(0)
#
## print values
#print(sheet.row_values(0))
