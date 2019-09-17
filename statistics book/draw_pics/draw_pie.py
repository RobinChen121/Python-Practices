#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/24 23:19
# @Author  : Zhen Chen

# Python version: 3.7
# Description: 画饼图

"""

import matplotlib.pyplot as plt
import numpy as np

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct


# 自定义的 autopct 函数
def func(pct, allvals):
    total = sum(allvals)
    absolute = int(round(pct*total/100.0))
    return "{:.1f}%\n({:d})".format(pct, absolute)


labels = ['果汁', '矿泉水', '绿茶', '其他', '碳酸饮料']
x = [10, 6, 11, 8, 15]
explode = [0, 0.1, 0, 0, 0]  # 突出显示第二个扇区

# plt.pie(x, explode=explode, labels=labels, autopct='%.2f%%', shadow=True, startangle=90)

plt.pie(x, explode=explode, labels=labels, autopct=lambda pct: func(pct, x),
        shadow=True, startangle=90)  # make_autopct(x),  # 利用 lambda 定义 pct 函数
plt.legend()  # 显示标签
plt.axis('equal')  # 让图形和坐标轴相等，这样饼图会更好看
plt.show()
