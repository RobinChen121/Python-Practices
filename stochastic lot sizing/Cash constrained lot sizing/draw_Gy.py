# -*- coding: utf-8 -*-
"""
@date: Created on Thu Aug 23 17:00:53 2018

@author: Zhen Chen

@Python version: 3.6

@descprition: draw G(y) for the last period

"""

import matplotlib.pyplot as plt
import scipy.stats as sp


meanDemand = 60;
iniCash = 110;
fixOrderCost = 100;
variCost = 1;
price = 8;
holdCost = 5;

Gy = []
max_demand = sp.poisson.ppf(0.99, meanDemand).astype(int)
for y in range(100):
    singleValue = 0
    for d in range(max_demand):
        revenue = price * min(y, d)
        Iplus = max(y - d, 0)
        ICost = holdCost * Iplus;
        orderCost = variCost * y
        cashIncre = revenue - ICost - orderCost
        singleValue -= sp.poisson.pmf(d, meanDemand) * cashIncre

    Gy.append(singleValue)
    
plt.ylim(-500, 0)
plt.plot(Gy)
