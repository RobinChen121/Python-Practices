# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:15:26 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: draw contour in python
    
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap  
import scipy


df = pd.read_csv(r'D:\Users\chen_\git\Stochastic-Inventory\H.csv', index_col=0) 
df.drop(df.columns[-1], axis=1, inplace = True)

df.columns = df.columns.astype(float)
R0 = df.index.values
x0 = df.columns.values

x, R = np.meshgrid(x0, R0)
GA = df.values.astype(float)

colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  
colors = ('lightgreen', 'lightblue')
cmap = ListedColormap(colors[:len(np.unique(GA))])  

plt.figure(dpi = 150) # 设置清晰度
#cp = plt.contourf(R, x, GA, 1)  # cmap = cmap
ax = plt.contour(R, x, GA, 0, colors='black', linewidths=1, linestyles='solid')

contour_line_values = np.stack(ax.allsegs[0])
x_values = contour_line_values[0][:, 0]
y_values = contour_line_values[0][:, 1]
d = y_values
plt.fill_between(x_values, y_values, interpolate=True, color='lightgreen')
plt.fill_between([27, 50], [6, 6],  facecolor = 'yellowgreen', hatch="/",edgecolor="k", linewidth = 0)

plt.plot([0, 50], [8, 8], 'r:')
plt.text(-1.5, 7.5, '$s$')
#plt.plot([0, 27.5], [5, 5], 'r:')
#plt.plot([27.4, 27.4], [0, 5], 'r:')
#plt.text(25.5, -1.5, '$C(x)$', fontsize=8)
#plt.colorbar(cp)

plt.xlabel('R')
plt.ylabel('x')
plt.savefig('H.eps', format='eps')
