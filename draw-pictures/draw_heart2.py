# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:43:06 2023

@author: chen
"""
import matplotlib.pyplot as plt
import numpy as np


# type %matplotlib qt to shown figure in a separate window

x = np.linspace(-2, 2, 1000)
alpha = 1

while alpha <= 21:
    plt.xlim(-3, 3)
    plt.ylim(-2, 4)
    y = abs(x)**(2/3) + 0.9*np.sqrt(3.3 - x**2)*np.sin(alpha*(np.pi)*x)
    plt.plot(x, y)
    
    plt.text(-1.6, 3, r'$f(x)=x^{2/3}+0.9(3.3-x^2)^{1/2}\sin(\alpha\pi x)$')   
    alpha_s = str(round(alpha, 2))
    tx = plt.text(-0.5, 2.5, r'$\alpha=$' + alpha_s)
    plt.pause(0.1) 
    if alpha <= 20:
        alpha += 0.1
        plt.clf()
    else:
        break
    
    