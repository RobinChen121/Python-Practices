# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 22:20:24 2018

@author: Zhen Chen

@Python version: 3.6

@description: draw a piecewise linear function
    
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib # for writing complex latex equations
matplotlib.rc('text', usetex = True)
matplotlib.rc('font', **{'family' : "sans-serif"})
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
plt.rcParams.update(params)

x = np.arange(0, 100, 1)
plt.plot(x, 300 + x)

x = np.arange(100, 200, 1)
plt.plot(x, 300 + 100 + 2 * (x-100))

x = np.arange(200, 300, 1)
plt.plot(x, 300 + 100 + 2 * (200 - 100) - 3 * (x-200))

plt.title(r'$f(x)=\begin{cases}300+x &x\leq 100\\300 + 100 + 2 (x-100)  & 100<x\leq 200\\300 + 100 + 2 * (200-100) - 3 (x-200)  & x>200\end{cases}$')

plt.xlim((0, 350)) # x scale
plt.ylim((100, 800))

plt.show()
