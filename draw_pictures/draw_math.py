"""
Created on 2025/2/5, 19:53 

@author: Zhen Chen.

@Python version: 3.10

@description:  

"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-1, 10, 0.1)

y = (1 + x)*np.exp(-x)

plt.plot(x, y, label = r'$y=(1+x)e^{-x}$')
plt.legend()
plt.show()