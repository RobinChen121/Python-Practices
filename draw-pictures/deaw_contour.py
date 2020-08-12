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

R, x = np.meshgrid(R0, x0)
G = df.values.astype(float)

colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')  
colors = ('white', 'lightgreen')
cmap = ListedColormap(colors[:len(np.unique(G))])  

plt.figure(figsize=(6.5,5), dpi = 50) # 设置清晰度
#cp = plt.contourf(x, R, GA, 0, cmap = cmap)  # cmap = cmap
ax = plt.contour(R, x, G.T, 0, colors='black', linewidths=1, linestyles='solid')

#contour_line_values = np.stack(ax.allsegs[0])
#x_values = contour_line_values[0][:, 0]
#y_values = contour_line_values[0][:, 1]
#d = y_values
#plt.fill_between(x_values, y_values, interpolate=True, color='lightgreen')
#plt.fill_between(np.arange(0, 50), np.ones(50), 30, interpolate=True, color='lightblue')
arr_x1 = ax.allsegs[1][0][0:8, 0]
arr_x2 = ax.allsegs[1][0][7:23, 0]
arr_y1 = ax.allsegs[1][0][0:8, 1]
arr_y2 = ax.allsegs[1][0][7:23, 1]
arr_y2_1 = 16.8 * np.ones((len(arr_x1)))
arr_y2_2 = 30 * np.ones((len(arr_x2)))
#plt.fill_between([0, 3], [30, 30], [16.8, 16.8], hatch="X", alpha=.99, facecolor="none", linewidth = 1)
#plt.fill_between(arr_x1, arr_y1, arr_y2_1, hatch="///", alpha=.99, facecolor="none", linewidth = 1)
#plt.fill_between(arr_x2, arr_y2, arr_y2_2, hatch="///", alpha=.99, facecolor="none", linewidth = 1)

arr = ax.allsegs[1][0]

#plt.text(-1.5, 7.5, '$s$', fontsize=8)
#plt.text(2.7, -1.5, '$x_0$', color = 'red')
#plt.plot([3, 3], [0, 16.8], 'r:')
#plt.plot(0, 16.8, 'ro', markersize = 3)
#plt.text(-0.7, 16.5, '$R_0$', color = 'red')
##plt.colorbar(cp)
#
plt.xlabel('R')
#plt.xticks([0, 5, 10, 15, 20])
plt.ylabel('x')
plt.savefig('J2.eps', format='eps')
