#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:06:43 2024

@author: zhenchen

@disp:  
    
    
"""

import numpy as np
import matplotlib.pyplot as plt


alpha = 0.2
x = np.arange(0, 50)
y1 = [(1 - alpha)**i for i in x]
y2 = [np.exp(-alpha*i) for i in x]

plt.plot(x, y1, label = r'$(1-\alpha)^i$')
plt.plot(x, y2, label = r'$e^{-\alpha i}$')
plt.title(r'$\alpha$ = ' + str(alpha) )
plt.legend()
plt.show()

plt.figure()
alpha = 0.5
x = np.arange(0, 50)
y1 = [(1 - alpha)**i for i in x]
y2 = [np.exp(-alpha*i) for i in x]

plt.plot(x, y1, label = r'$(1-\alpha)^i$')
plt.plot(x, y2, label = r'$e^{-\alpha i}$')
plt.title(r'$\alpha$ = ' + str(alpha) )
plt.legend()
plt.show()

plt.figure()
alpha = 0.8
x = np.arange(0, 50)
y1 = [(1 - alpha)**i for i in x]
y2 = [np.exp(-alpha*i) for i in x]

plt.plot(x, y1, label = r'$(1-\alpha)^i$')
plt.plot(x, y2, label = r'$e^{-\alpha i}$')
plt.title(r'$\alpha$ = ' + str(alpha) )
plt.legend()
plt.show()