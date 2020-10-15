#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/9/11 23:16
# @Author  : Zhen Chen

# Python version: 3.7
# Description:  fit a gamma distribution
# 伽马分布是一个右偏分布

"""

from fitter import Fitter
from scipy import stats
import matplotlib.pyplot as plt

data = stats.gamma.rvs(a = 2, loc=10, scale=5, size=10000) # 生成随机数，2 表示伽马分布的 shape parameter： alpha, scale 就是伽马分布的 scale 参数

mean, var, skew, kurt = stats.gamma.stats(a = 2, loc=10, scale=5, moments='mvsk')

f = Fitter(data, distributions=['gamma'])
f.fit()
# may take some time since by default, all distributions are tried
# but you call manually provide a smaller set of distributions
f.summary()