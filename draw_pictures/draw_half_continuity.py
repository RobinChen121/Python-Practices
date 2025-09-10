# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:38:04 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: 
    
"""
import matplotlib.pyplot as plt
import numpy as np


ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
# Create 'x' and 'y' labels placed at the end of the axes
ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)
# Draw arrows
arrow_fmt = dict(markersize=4, color='black', clip_on=False)
ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)

x1 = np.arange(0, 3, 0.1)
y1 = np.zeros((30))
plt.plot(x1, y1)
x2 = np.arange(-3, 0, 0.1)
y2 = np.ones((30))
plt.plot(x2, y2)
plt.scatter(0, 1, marker = 'o', c = 'r', edgecolor = 'r', s = 50)
plt.xticks(np.arange(-3, 3, 1))
plt.yticks(np.arange(-1, 3))
plt.show()

