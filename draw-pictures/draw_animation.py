#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:02:16 2023

@author: zhenchen

@disp:  
    
    
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

fig = plt.figure()
sub = fig.add_subplot(111,xlim=(-3, 3), ylim=(-2, 4))
PLOT, = sub.plot([],[])
time_text = sub.text(1,1,"",transform = sub.transAxes, ha="right")

def init():
    PLOT.set_data([],[])
    time_text.set_text("")
    return PLOT,time_text

def animate(i):
    x = np.linspace(-2, 2, 1000)
    y = np.abs(x)**(2/3) + 0.9*np.sqrt(3.3 - x**2)*np.sin(i*(np.pi)*x)

    PLOT.set_data(x,y)
    return PLOT

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=2000, interval=20, blit=True)

plt.show()