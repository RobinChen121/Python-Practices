# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:43:39 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description:  draw prob plot, not same with pp and qq plot, get a best fit line
    
"""

import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np


n = 100
samples = st.norm.rvs(loc = 5, scale = 2, size = n)

samples_sort = sorted(samples)


x_labels_p = np.zeros(n)
x_labels_p[0] = 1 - 0.5 ** (1/n)
x_labels_p[n - 1] = 0.5 ** (1/n)
for i in range(1, n - 1):
    x_labels_p[i] = (i + 1 - 0.3175)/(n + 0.365)
y_labels_p = st.norm.cdf(samples_sort, loc = 5, scale = 2)

plt.scatter(x_labels_p, y_labels_p)
plt.title('PP plot for normal distribution samle')
plt.show()


x_labels_q = samples_sort
y_labels_q = st.norm.ppf(x_labels_p, loc = 5, scale = 2)

plt.scatter(x_labels_q, y_labels_q)
plt.title('QQ plot for normal distribution samle')
plt.show()

res = st.probplot(samples, sparams=(5, 2), plot = plt) # 若没有 sparams，默认会标准化样本数据
plt.title('QQ plot by probplot for normal distribution')
plt.show()

## poisson distribution
samples2 = st.poisson.rvs(mu = 5, size = n)

samples_sort2 = sorted(samples2)

n = len(samples)
x_labels_p = np.zeros(n)
x_labels_p[0] = 1 - 0.5 ** (1/n)
x_labels_p[n - 1] = 0.5 ** (1/n)
for i in range(1, n -1):
    x_labels_p[i] = (i + 1 - 0.3175)/(n + 0.365)
y_labels_p = st.poisson.cdf(samples_sort2, mu = 5)

plt.scatter(x_labels_p, y_labels_p)
plt.title('PP plot for poisson distribution sample')
plt.show()


x_labels_q = samples_sort
y_labels_q = st.poisson.ppf(x_labels_p, mu = 5)

plt.scatter(x_labels_q, y_labels_q)
plt.title('QQ plot for poisson distribution sample')
plt.show()

res2 = st.probplot(samples2, dist = st.poisson, sparams=(5), plot = plt) # 默认会标准化
plt.title('QQ plot by probplot for poisson distribution')
