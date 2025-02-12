# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:16:09 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap  


df = pd.read_csv(r'D:\Users\chen_\git\Stochastic-Inventory\H.csv', index_col=0) 
df.drop(df.columns[-1], axis=1, inplace = True)

df.columns = df.columns.astype(float)
R0 = df.index.values
x0 = df.columns.values

x, R = np.meshgrid(x0, R0)
GA = df.values.astype(float)

colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  
colors = ('white', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(GA))])  

plt.figure(figsize=(6.5,5), dpi = 300) # 设置清晰度
#cp = plt.contourf(x, R, GA, 0, cmap = cmap)  # cmap = cmap
ax = plt.contour(x, R, GA, 0, colors='black', linewidths=1, linestyles='solid')


arr_x = ax.allsegs[1][0][:, 0]
plt.fill_between(arr_x, ax.allsegs[1][0][:, 1], 30 * np.ones((len(arr_x))), hatch="///", alpha=.99, facecolor="none", linewidth = 1)


plt.plot([5.8, 5.8], [0, 30], 'r:')
plt.text(5.7, -1, '$s$', color = 'red')
plt.plot([3, 3], [0, 17], 'r:')
plt.plot([0, 3], [16.8, 16.8], 'r:')
plt.text(2.7, -1, '$x_0$', color = 'red')
plt.text(-1.5, 16.5, '$C(x_0)$', fontsize = 10, color = 'red')

##plt.colorbar(cp)
#
plt.xlabel('x')
plt.xticks([0, 5, 10, 15, 20])
plt.ylabel('R')
plt.savefig('J2.eps', format='eps')
