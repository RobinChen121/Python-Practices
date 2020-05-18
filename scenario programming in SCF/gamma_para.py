# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:22:31 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

import numpy as np
import scipy.stats as st
import pandas as pd


item_id = 591120591607 # 4571301 #
address = 'E:\爬虫练习\电商评论\月需求-商品' + str(item_id) + '-' + str(datetime.date.today()) + '.xlsx'

xls_read = pd.read_excel(address)

history_data = xls_read['demand']

mean_demand = np.average(history_data)
var_demand = np.var(history_data)

scale = var_demand / mean_demand
shape = mean_demand / scale
skew = st.skew(history_data)
kurt = st.kurtosis(history_data)
print(f'shape is {shape: .2f}')
print(f'scale is {scale: .2f}')
print(f'mean is {mean_demand: .2f}')
print(f'variance is {var_demand: .2f}')
print(f'skew is {skew: .2f}')
print(f'kurt is {kurt: .2f}')