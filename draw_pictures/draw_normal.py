# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:04:27 2018

@author: Zhen Chen

@Python version: 3.6

@description: draw a normal distribution for a hyper plane
    
"""
# import mpmath as mp
#
#
# mp.splot(lambda x, y: 1/4*(10-2*x-3*y))


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 参数设置
mu = 10000    # 均值
sigma = 2000  # 标准差

# 生成 x 轴数据，覆盖正态分布的主要范围（均值 ± 4 个标准差）
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)  # 从 2000 到 18000

# 计算正态分布的概率密度函数 (PDF)
y = norm.pdf(x, mu, sigma)  # 使用 scipy 的 norm.pdf 计算

# 绘制曲线
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(x, y, 'b-', label=f'Normal PDF (μ={mu}, σ={sigma})')  # 蓝色实线
plt.fill_between(x, y, color='skyblue', alpha=0.4)  # 填充整个曲线下的区域
plt.axhline(0, color='black', linewidth=0.5)  # x 轴
# plt.axvline(mu, color='red', linestyle='--', label=f'μ = {mu}')  # 均值线
plt.fill_between(x, y, where=(x >= mu - 4*sigma) & (x <= 12453), color='skyblue')  # 填充整个曲线下的区域
plt.axvline(12453, color='red', linestyle='--', label='0.89 quantile')  # 上界

# 添加标签和标题
plt.xlabel('x')
plt.ylabel('F(Q)')
plt.title('Normal Distribution PDF (Mean=10000, Std=2000)')
plt.legend()

# 设置 y 轴范围以清晰显示
max_density = norm.pdf(mu, mu, sigma)  # 峰值密度
plt.ylim(-0.01 * max_density, max_density * 1.2)
plt.xlim(2000, 18000)

# 显示图形
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()