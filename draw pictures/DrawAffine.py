# -*- coding: utf-8 -*-
"""
@date: Created on Fri Jul 27 19:54:42 2018

@author: Zhen Chen

@Python version: 3.6

@descprition:

"""

import matplotlib.pyplot as plt

plt.arrow(2, 3, 4, 8, width = 0.08, color = 'c', length_includes_head = True)

plt.plot(2, 3, 'ro')
plt.annotate('$x_2$', xy = (2.2, 2.8))
plt.plot(5, 9, 'ro')
plt.annotate('$x_1$', xy = (5.2, 8.8))
plt.plot(6, 11, 'ro')
plt.annotate('$y=x_2+\\theta (x_1-x_2), \\theta = 2$', xy = (6.2, 10.8))
plt.xlim((0, 10)) # x scale
plt.ylim((0, 16))
# or plt.axis([0, 10, 0, 16])
plt.show()
