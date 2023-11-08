#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:35:14 2023

@author: zhenchen

@disp:  
    
    
"""


# type %matplotlib qt to shown figure in a separate window

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

fig = plt.figure()
sub = fig.add_subplot(111,xlim=(-3, 3), ylim=(-2, 4))
PLOT,  = sub.plot([],[])
plt.text(-1.6, 3, r'$f(x)=x^{2/3}+0.9(3.3-x^2)^{1/2}\sin(\alpha\pi x)$') 



def animate(alpha):
    alpha_s = str(round(alpha, 2))
    # tx = plt.text(-0.5, 2.5, r'$\alpha=$' + alpha_s)
    x = np.linspace(-2,2,1000) 
    y = abs(x)**(2/3)+ 0.9*np.sqrt(3.3 - x**2)*np.sin(alpha*(np.pi)*x)

    PLOT.set_data(x,y)
    return PLOT, 

ani = animation.FuncAnimation(fig, animate, frames=np.arange(1, 20, 0.1), interval=100)

plt.show()