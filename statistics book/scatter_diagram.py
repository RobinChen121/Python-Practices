#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/18 0:35
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description: draw scatter diagram in Python
               在 python 中画散点图

"""

import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

gdp_rate = [9.40, 10.60, 9.60, 7.90, 7.80, 7.30, 6.90, 6.70, 6.80, 6.60]
first_industry_rate = [4.00, 4.30, 4.20, 4.50, 3.80, 4.10, 3.90, 3.30, 4.00, 3.50]
years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]

plt.plot(years, gdp_rate, '.-')
plt.plot(years, first_industry_rate)
plt.xticks(years)  # 横坐标显示给定的年份
plt.xlabel('年份')
plt.ylabel('GDP 增长率')
plt.show()