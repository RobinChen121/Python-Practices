#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/9/6 13:35
# @Author  : Zhen Chen

# Python version: 3.7
# Description: general scenarios for a given demand distribution based on the
               methods proposed by Heitsch & Romisch in "Scenario reduction algorithms in stochastic
               programming" (2013) of Computational Optimization and Application
               
               results not good by scipy, may not be feasible

"""

from scipy.optimize import minimize
from scipy.optimize import Bounds
import numpy as np
from scipy.optimize import LinearConstraint


def objective(x):
    T = int(len(x) / 2)
    demand = x[1:T]
    possibility = x[T + 1:2 * T]
    sample_mean = np.dot(demand, possibility)
    sample_variance = 0
    for i in range(T):
        sample_variance += x[T + i]*(x[i] - sample_mean)**2
    sample_skew = 0;
    for i in range(T):
        sample_skew += x[T + i]*(x[i] - sample_mean)**3 / sample_variance **1.5
    sample_kurt = 0
    for i in range(T):
        sample_kurt += x[T + i]*(x[i] - sample_mean)**4 / sample_variance **2

    obj = 0.25 * (sample_mean - mean)**2 + 0.25 * (sample_variance - variance)**2 + \
             0.25 * (sample_skew - skew)**2 + 0.25 * (sample_kurt - kurt)**2
    return obj


# three demands, all follow Weibull distibution
global mean 
global variance
global skew
global kurt
mean = 82.14
variance = 5173.98
skew = 2.06
kurt = 4.39

x0 = [mean/5, mean/5, mean/5, mean/5, mean/5, 0.2, 0.2, 0.2, 0.2, 0.2]
linear_constraint = LinearConstraint([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 1, 1)
bounds = [(1, 10000), (1, 10000), (1, 10000), (1, 10000), (1, 10000), (0.1, 0.35), (0.1, 0.35), (0.1, 0.35), (0.1, 0.35), (0.1, 0.35)]
res = minimize(objective, x0, method='nelder-mead', constraints = linear_constraint, bounds = bounds, options={'xtol': 1e-8, 'disp': True})
print(res.x)
