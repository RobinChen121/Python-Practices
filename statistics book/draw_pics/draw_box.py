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

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

marks = [[76, 90, 97, 71, 70, 93, 86, 83, 78, 85, 81],
         [93, 81, 76, 88, 66, 79, 83, 92, 78, 86, 78],
         [74, 87, 85, 69, 90, 80, 77, 84, 91, 74, 70],
         [68, 75, 70, 84, 73, 60, 76, 81, 88, 68, 75],
         [70, 73, 92, 65, 78, 87, 90, 70, 66, 79, 68],
         [55, 91, 68, 73, 84, 81, 70, 69, 94, 62, 71]]
courses = ('英语', '西方经济学', '市场营销学', '财务管理', '基础会计学', '统计学')

plt.boxplot(marks, showfliers=False)
plt.xticks(np.arange(1, 7), courses)
plt.show()
