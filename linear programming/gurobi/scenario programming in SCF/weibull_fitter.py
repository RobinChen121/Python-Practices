# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:24:23 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: fit a weibull distribution
# weibull_min 是一个右偏分布
    
"""
from fitter import Fitter
from scipy import stats
import matplotlib.pyplot as plt

data = stats.weibull_min.rvs(0.8114516645976357, 39.99999999999999, 143.51883068442237, size=10000) 

f = Fitter(data, distributions=['weibull_min'])
f.fit()
f.summary()