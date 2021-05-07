# -*- coding: utf-8 -*-
"""
@date: Created on Fri Jul 27 11:16:59 2018

@author: Zhen Chen

@Python version: 3.6
 
@descprition:
     
"""


import matplotlib.pyplot as plt
import numpy as np


plt.ylim(0,1)
plt.xlim(-0.1,1)

x1 = np.arange(-10, 0, 0.001)
x2 = np.arange(0, 10, 0.001)
plt.plot(x2, 0.63*x2+0.128, 'b')
plt.plot(x1, 0.391*x1+0.128, 'b')

x3 = np.arange(-10, 0, 0.1)
x4 = np.arange(0, 10, 0.1)
plt.plot(x4, 0.63*x4+0.116, 'r')
plt.plot(x3, 0.502*x3+0.116, 'r')


plt.show()

