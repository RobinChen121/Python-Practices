# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 22:01:16 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""
import pandas as pd


ini_data = pd.read_excel(r'D:\项目\电网\2021\old\采集数据（清理版）.xlsx', index_col = 0)
clean_data = ini_data.dropna() # ini_data.fillna(0) 
clean_data['sum'] = ini_data.iloc[:, 2:].sum(axis = 1)
sum_data = clean_data[['户号对应', '日期', 'sum']]
sum_data.to_excel('sumCleanData.xlsx', index = False)
