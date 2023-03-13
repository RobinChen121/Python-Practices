#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 22:04:46 2023

@author: chen
"""

import pandas as pd

sale_datas = pd.read_csv(r'/Users/chen/Documents/投稿/在写论文/杉树科技竞赛/original_data/sales_data.csv')
# output to single csv file for each sku
df1 = sale_datas.groupby('sku_id')
dfs = []
for name, data in df1:
    dfs.append(data)
    file_address = '/Users/chen/Documents/投稿/在写论文/杉树科技竞赛/sku_sales_data/'
    xls_name = file_address + str(name) + '.csv'
    data.to_csv(xls_name, index = False)

# output to single csv file for each idx
df2 = sale_datas.groupby('idx')
dfs = []
for name, data in df2:
    dfs.append(data)
    file_address = '/Users/chen/Documents/投稿/在写论文/杉树科技竞赛/dc_sku_sales_data/'
    xls_name = file_address + str(name) + '.csv'
    data.to_csv(xls_name, index = False)

leadtime = pd.read_csv(r'/Users/chen/Documents/投稿/在写论文/杉树科技竞赛/original_data/leadtime_data.csv')
df3 = leadtime.groupby('idx')
dfs = []
for name, data in df3:
    dfs.append(data)
    file_address = '/Users/chen/Documents/投稿/在写论文/杉树科技竞赛/dc_sku_leadtime_data/'
    xls_name = file_address + str(name) + '.csv'
    data.to_csv(xls_name, index = False)