#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:35:14 2023

@author: zhen chen

@disp:  
    
    
"""


# type %matplotlib qt to shown figure in a separate window

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(alpha):
    x = np.linspace(-1.8,1.8,1000)
    y = abs(x)**(2/3) + 0.9*np.sqrt(3.3 - x**2)*np.sin(alpha*(np.pi)*x)
    PLOT.set_data(x, y)
    time_text.set_text(r'$\alpha$ = ' + str(round(alpha, 2)))
    return PLOT, time_text

fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-2.5, 2.5), ylim=(-2, 4)) # or plt.subplot
PLOT, = ax.plot([], []) # return all the lines
plt.text(-1.2, 3, r'$f(x)=x^{2/3}+0.9(3.3-x^2)^{1/2}\sin(\alpha\pi x)$') 
time_text = ax.text(-0.45, 2.5,'') # transform = ax.transAxes

ani = FuncAnimation(fig, animate, frames = 100, interval = 200, repeat = False)
plt.show()
# ani.save("heart.gif") # 保存图像为 1 个 gif 文件