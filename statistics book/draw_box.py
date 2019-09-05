#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/25 18:10
# @Author  : Zhen Chen

# Python version: 3.7
# Description: 画箱线图

"""
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

# fake up some data
spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = spread

fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(data)
plt.show()