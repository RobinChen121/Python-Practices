#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/21 17:11
# @Author  : Zhen Chen

# Python version: 3.7
# Description: 画出近十年我国 GDP 增长率的趋势图

"""

import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入纵坐标轴数据与横坐标轴数据
gdp_rate = [9.40, 10.60, 9.60, 7.90, 7.80, 7.30, 6.90, 6.70, 6.80, 6.60]
first_industry_rate = [4.00, 4.30, 4.20, 4.50, 3.80, 4.10, 3.90, 3.30, 4.00, 3.50]
second_industry_rate = [10.30, 12.70, 10.70, 8.40, 8.00, 7.40, 6.20, 6.30, 5.90, 5.80]
third_industry_rate = [9.60, 9.70, 9.50, 8.00, 8.30, 7.80, 8.20, 7.70, 7.90, 7.60]
years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]

# 4 个 plot 函数画出 4 条线，线形为折线，每条线对应各自的标签 label
plt.plot(years, gdp_rate, '.-', label='GDP增长率')
plt.plot(years, first_industry_rate, '.-', label='第一产业增长率')
plt.plot(years, second_industry_rate, '.-', label='第二产业增长率')
plt.plot(years, third_industry_rate, '.-', label='第三产业增长率')

plt.xticks(years)  # 设置横坐标刻度为给定的年份
plt.xlabel('年份') # 设置横坐标轴标题
plt.legend() # 显示图例，即每条线对应 label 中的内容
plt.show() # 显示图形


# plt.plot(years, gdp_rate, '.-', label = 'GDP增长率', years, first_industry_rate, '.-', label='第一产业增长率')