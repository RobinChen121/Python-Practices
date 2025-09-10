# -*- coding: utf-8 -*-
"""
@date: Created on Fri Jul 27 11:16:59 2018

@author: Zhen Chen

@Python version: 3.6
 
@description:
     
"""


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.1)
plt.plot(x, np.sin(x)/x)

# facecolor generates a hollow circle, s is the size
plt.scatter(0, 1, facecolor = 'none', edgecolor = 'r', s = 100)
plt.show()

