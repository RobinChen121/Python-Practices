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
from pylab import savefig

df = pd.read_csv(r'D:\Users\chen_\git\Stochastic-Inventory\G.csv', index_col=0)
ax = plt.subplots(figsize=(15, 15), dpi = 120) 
ax = sns.heatmap(df, cmap='GnBu', annot=True, fmt='.3g', cbar=False)
ax.set_xlabel('x', fontsize=15, labelpad = 15)
ax.set_ylabel('R', fontsize=15, labelpad = 15)

figure = ax.get_figure()    
figure.savefig('Gvalues.eps', dpi=200)

df2 = pd.read_excel(r'D:\Users\chen_\git\Statistics-book\datas\data-scatter.xlsx')
df3 = df2.pivot(index = '产量', columns = '温度', values = '降雨量')



