#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 22:23:11 2023

@author: zhen chen

@disp:  
    
    
"""

import matplotlib.pyplot as plt
import numpy as np
import math

x = np.linspace(-2, 2, 1000)
y1 = [math.sqrt(2*abs(i) - i**2) for i in x]
y2 = [-2.14*math.sqrt(math.sqrt(2)-math.sqrt(abs(i))) for i in x]

plt.plot(x, y1, 'r')
plt.plot(x, y2, 'r')
plt.fill_between(x, y1, y2, color='r')

plt.show()