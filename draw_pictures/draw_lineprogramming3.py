"""
Created on 2025/2/8, 21:17 

@author: Zhen Chen.

@Python version: 3.10

@description:  

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import linprog

import matplotlib
matplotlib.use('TkAgg')

# 定义优化问题
c = [-2, -3]
A = [[2, 1], [1, 2]]
b = [4, 5]
res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)], method='highs')

# 绘图
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.grid(color='gray', linestyle='--', linewidth=0.5)

x1 = np.linspace(0, 5, 400)
x2_1 = (4 - 2 * x1) / 1
x2_2 = (5 - 1 * x1) / 2
ax.plot(x1, x2_1, label=r'$2x_1 + x_2 \leq 4$', color='blue')
ax.plot(x1, x2_2, label=r'$x_1 + 2x_2 \leq 5$', color='green')
ax.fill_between(x1, np.minimum(x2_1, x2_2), 0, where=(x2_1 > 0) & (x2_2 > 0), color='lightgrey', alpha=0.5)
line, = ax.plot([], [], 'r-', label='Objective Function')
ax.legend()

def update(c):
    x2 = (c - 2 * x1) / 3
    line.set_data(x1, x2)
    return line,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 10, 100), blit=True, interval=50)
plt.show()
