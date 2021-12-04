# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:45:02 2021

@author: zhen chen

MIT Licence.

Python version: 3.8


Description:  对2021年第二次给的数据，按第五级行业提取
    
"""

import pandas as pd


ini_data = pd.read_excel(r'D:\项目\电网\2021\new\LJSD历史数据加行业清理版.xlsx', index_col = 0)
clean_data = ini_data.dropna() # ini_data.fillna(0) 
ini_data['sum'] = ini_data.iloc[:, 2:25].sum(axis = 1)
sum_data = ini_data[['编号', '数据日期', '一级行业', '二级行业', '三级行业', '四级行业', '五级行业', 'sum']]
sum_data.to_excel('sumCleanData.xlsx', index = False)