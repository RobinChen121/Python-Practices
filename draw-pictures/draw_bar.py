# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:47:18 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

import matplotlib.pyplot as plt

 # 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # ['SimHei'] for windows
# plt.rcParams['axes.unicode_minus'] = False

waters = ['碳酸饮料', '绿茶', '矿泉水', '果汁','其他']
buy_number = [6, 7, 6, 1, 2]

plt.bar(waters, buy_number)  # 横放条形图函数 barh
plt.title('男性购买饮用水情况的调查结果')

plt.show()
