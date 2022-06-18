# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:33:13 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: 
    
"""
import pandas as pd


datas = pd.read_excel(r"D:\项目\电网\2022\0505-1.xlsx")
datas['day_consuming'] = datas.iloc[:, 2:26].sum(axis = 1)
df = datas[['cons_no', 'data_date', 'day_consuming']]

# 将所有数据输出到单独的 excel 表
# df['group'] = df['cons_no'].ne(df['cons_no'].shift()).cumsum()
df1 = df.groupby('cons_no')
dfs = []
for name, data in df1:
    dfs.append(data)
    file_address = 'D:\\项目\\电网\\2022\\各用户用电量\\'
    xls_name = file_address + '用户' + str(name) + '.xlsx'
    data.to_excel(xls_name, index = False)