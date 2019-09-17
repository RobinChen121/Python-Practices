#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/9/6 13:35
# @Author  : Zhen Chen

# Python version: 3.7
# Description: general scenarios for a given demand distribution based on the
               methods proposed by Heitsch & Romisch in "Scenario reduction algorithms in stochastic
               programming" (2013) of Computational Optimization and Application

"""

from scipy.optimize import minimize
from scipy.optimize import Bounds
import numpy as np


def objective(x):
    T = int(len(x) / 2)
    demand = x[1:T]
    possibility = x[T + 1:2 * T]
    sample_mean = np.dot(demand, possibility)
    sample_variance = 0
    for i in range(T):
        sample_variance += x[T + i]*(x[i] - sample_mean)**2

    obj = 0.5 * (sample_mean - mean[0])**2 + 0.5 * (sample_variance - variance[0])**2
    return obj


# three demands, all follow Weibull distibution
global mean
mean = [467.25, 33.82, 149.7]
global variance
variance = [99.422, 175.4231, 4877.8]
skew = [1.06, 0.25, 0.47]
kurt = [4.35, 2.78, 2.98]

x0 = [3, 6, 9, 0.3, 0.3, 0.6]
res = minimize(objective, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
print(res.x)
