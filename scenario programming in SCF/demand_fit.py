#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/9/10 20:31
# @Author  : Zhen Chen

# Python version: 3.7
# Description:  fit the best distribution given a list of history data, 
it seems gamma distribution is appropriate for one tmall sales data

"""

from fitter import Fitter
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

history_data = [141, 159, 166, 172, 177, 182, 188, 196, 203, 214,
                143, 160, 167, 173, 177, 183, 189, 196, 203, 215,
                144, 160, 168, 173, 178, 184, 189, 196, 205, 218,
                149, 161, 168, 174, 178, 185, 189, 196, 206, 223,
                150, 161, 168, 174, 178, 186, 190, 196, 207, 225,
                152, 162, 170, 174, 179, 186, 190, 197, 208, 226,
                153, 163, 171, 175, 179, 187, 191, 197, 209, 228,
                153, 163, 171, 175, 179, 187, 192, 198, 210, 233,
                154, 164, 172, 175, 180, 187, 194, 198, 210, 233,
                155, 165, 172, 175, 180, 187, 194, 200, 211, 234,
                156, 165, 172, 176, 181, 188, 195, 201, 211, 234,
                158, 165, 172, 176, 182, 188, 195, 202, 213, 237]

histroy_data = [90, 150, 610, 240, 310, 170,  40, 110,  60,  50,  80, 270]

# 从 2017年8月到2020年3月的需求推测
history_data = [10,30,150,90,120,320,40,410,170,170,60,60,150,90,150,330,100,60,70,60,100,90,210,130,140,40,40,110,60,40,20,20]

history_data = [140,130,120,110,100,100,90,90,90,70,60,60,60,60,60,40,40,40,40] # 中需求：50~150
#history_data = [410,330,320,300,210,170,170, 150,150] # 高需求 >150
#history_data = [40,40,40,40,30,20,20,10] # 低需求<50


print(len(history_data))

f = Fitter(history_data, distributions=['expon', 'norm', 'beta', 'gamma', 'weibull_min'])
f.fit()
print(f.fitted_param)

# may take some time since by default, all distributions are tried
# but you call manually provide a smaller set of distributions
f.summary()

# show histograms, mean, variance, skew, kurt
# the histogram of the data
# n, bins, patches = plt.hist(history_data, edgecolor='k', alpha=0.35) # 设置直方边线颜色为黑色，不透明度为 0.35

mean = np.mean(history_data)
print('mean is %.2f' % mean)
variance = np.var(history_data)
std = np.sqrt(variance)
print(' std is %.2f' % std)
skew = stats.skew(history_data)
print('skew is %.2f' % skew)
kurt = stats.kurtosis(history_data)
print('kurt is %.2f' % kurt)

plt.show()
