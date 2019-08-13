# -*- coding: utf-8 -*-
"""
@date: Created on Fri Jul 27 11:16:59 2018

@author: Zhen Chen

@Python version: 3.6
 
@descprition:
     
"""


import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.1)
plt.plot(x, np.sin(x)/x)

# c is center color, '' generates a hollow circle, s is the size
plt.scatter(0, 1, marker = 'o', c = '', edgecolor = 'r', s = 50)
plt.show()

