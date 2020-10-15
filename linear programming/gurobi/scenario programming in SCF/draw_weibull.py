#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/9/12 13:31
# @Author  : Zhen Chen

# Python version: 3.7
# Description:  draw a weibull distribution pdf. The weibull distribution belongs
                to the family of Exponentiated Weibull distribution

                pdf of weibull distribution is f(x) = (a/k)*(x/k)^(a-1)*exp((-x/k)^a)
                when a = 1, weibull distribution is exponential distribution



"""

import numpy as np
import matplotlib.pyplot as plt


# define the pdf of weibull distribution
def weib(x, scale, shape):
    return (shape / scale) * (x / scale)**(shape - 1) * np.exp(-(x / scale) ** shape)


scale = 50
shape = 1.5
x = np.arange(1, scale*2)
y = np.zeros(len(x))  # [0 for i in range(len(x))]
for i in range(len(x)):
    y[i] = weib(x[i], scale, shape)

scale = 50
shape = 2.5
y1 = np.zeros(len(x))  # [0 for i in range(len(x))]
for i in range(len(x)):
    y1[i] = weib(x[i], scale, shape)
scale = 50
shape = 4
y2 = np.zeros(len(x))  # [0 for i in range(len(x))]
for i in range(len(x)):
    y2[i] = weib(x[i], scale, shape)
print(np.mean(y))
print(np.mean(y1))
print(np.mean(y2))
scale = 30
shape = 2.5
y3 = np.zeros(len(x))  # [0 for i in range(len(x))]
for i in range(len(x)):
    y3[i] = weib(x[i], scale, shape)
scale = 70
shape = 2.5
y4 = np.zeros(len(x))  # [0 for i in range(len(x))]
for i in range(len(x)):
    y4[i] = weib(x[i], scale, shape)


plt.subplot(2, 1, 1)
plt.plot(x, y, 'r', label='scale=50, shape=1.5')
plt.plot(x, y1, 'b', label='scale=50, shape=2.5')
plt.plot(x, y2, 'g', label='scale=50, shape=4')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x, y3, 'r', label='scale=30, shape=2.5')
plt.plot(x, y1, 'b', label='scale=50, shape=2.5')
plt.plot(x, y4, 'g', label='scale=70, shape=2.5')
plt.legend()
plt.show()
