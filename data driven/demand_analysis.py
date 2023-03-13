#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:53:16 2023

@author: chen
"""

import pandas as pd



file_name = 'SKU001.csv'
df = pd.read_csv('/Users/chen/Documents/投稿/在写论文/杉树科技竞赛/sku_sales_data/' + file_name)
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d') 
df2 = df.groupby('date').sum()