# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:43:06 2023

@author: chen
"""
import matplotlib.pyplot as plt
import numpy as np
import time

alpha =  1
x = np.linspace(-2, 2
                , 100)

alpha = 20
for i in range(1, 50):
    alpha += 0.1*i
    y = abs(x)**(2/3) + 0.9*np.sqrt(3.3 - x**2)*np.sin(alpha*(np.pi)*x)
    plt.plot(x, y, 'b')
    plt.pause(0.5)
    # plt.clf()
    
    