# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 16:31:16 2022

@author: chen
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

n = 50
p = 0.1

x = np.arange(1, 3*n*p)
y = [st.binom.pmf(i, n, p) for i in x]
plt.plot(x, y, 'r', label='binormial, n=50,p=0.1')
plt.legend()

y1 = [st.norm.pdf(i, n*p, np.sqrt(n*p*(1-p))) for i in x]
plt.plot(x, y1, 'b', label='norm, mu=np, sigma^2=np(1-p)')
plt.legend(fontsize=5)


plt.figure()
n = 100
p = 0.1

x = np.arange(1, 3*n*p)
y = [st.binom.pmf(i, n, p) for i in x]
plt.plot(x, y, 'r', label='binormial, n=100,p=0.1')
plt.legend()

y1 = [st.norm.pdf(i, n*p, np.sqrt(n*p*(1-p))) for i in x]
plt.plot(x, y1, 'b', label='norm, mu=np, sigma^2=np(1-p)')
plt.legend(fontsize=5)