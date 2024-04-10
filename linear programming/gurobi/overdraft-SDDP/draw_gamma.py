#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:10:47 2024

@author: zhenchen

@disp:  
    
    
"""

import matplotlib.pyplot as plt
import scipy.special as sps  
import numpy as np

# gamma distribution:mean demand is shape / beta and variance is shape / beta^2
# beta = 1 / scale
# shape = demand * beta
# variance = demand / beta
meanDemands =[30, 10] # higher average demand vs lower average demand
betas = [20, 2] # lower variance vs higher variance



shape, scale = meanDemands[0]*betas[0], 1/betas[0]
s = np.random.gamma(shape, scale, 1000)
count, bins, ignored = plt.hist(s, 50, density=True)
y = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')  
plt.show()

shape, scale = meanDemands[1]*betas[1], 1/betas[1]
s = np.random.gamma(shape, scale, 1000)
count, bins, ignored = plt.hist(s, 50, density=True)
y = bins**(shape-1)*(np.exp(-bins/scale) /  
                      (sps.gamma(shape)*scale**shape))
plt.plot(bins, y, linewidth=2, color='r')  
plt.show()