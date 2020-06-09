# -*- coding: utf-8 -*-
"""
Created on Mon May 18 22:06:53 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: draw heatmap by seaborn
    
""" 

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel(r'resultTable.xls', index_col=0)
fig, ax = plt.subplots(figsize=(15, 15)) 
ax = sns.heatmap(df, cmap='GnBu', annot=True)
ax.set_xlabel('R', fontsize=15, labelpad = 15)
ax.set_ylabel('x', fontsize=15, labelpad = 15)

df2 = pd.read_excel(r'D:\Users\chen_\git\Statistics-book\datas\data-scatter.xlsx')
df3 = df2.pivot(index = '产量', columns = '温度', values = '降雨量')



