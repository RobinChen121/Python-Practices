# -*- coding: utf-8 -*-
"""
Created on Fri May 22 22:30:23 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

n = 100
samples = st.norm.rvs(loc = 5, scale = 2, size = n)

samples_sort = sorted(samples)

x_labels_p = np.arange(1/(2*n), 1, 1/n)
y_labels_p = st.norm.cdf(samples_sort, loc = 5, scale = 2)

plt.scatter(x_labels_p, y_labels_p)
plt.title('PP plot for normal distribution')
plt.show()


x_labels_q = samples_sort
y_labels_q = st.norm.ppf(x_labels_p, loc = 5, scale = 2)

plt.scatter(x_labels_q, y_labels_q)
plt.title('QQ plot for normal distribution')
plt.show()

import statsmodels.api as sm
probplot = sm.ProbPlot(samples, dist = st.norm, loc = 5, scale = 2)
probplot.qqplot(line='45')

#res = st.probplot(samples, sparams=(5, 2), plot = plt) # 若没有 sparams，默认会标准化样本数据
#plt.title('QQ plot by probplot for normal distribution')
#plt.show()



