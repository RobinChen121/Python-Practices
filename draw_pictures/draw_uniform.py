"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/3/5 15:55
Description: 
    

"""
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
a = 40  # 均匀分布的下界
b = 60  # 均匀分布的上界
density = 1 / (b - a)  # 概率密度值，1 / (b - a)

# 生成 x 轴数据，稍微扩展范围以显示完整曲线
x = np.linspace(a - 10, b + 10, 1000)  # 从 30 到 70 生成 1000 个点

# 计算概率密度函数 (PDF)
y = np.where((x >= a) & (x <= b), density, 0)  # 在 [40, 60] 内为 density，其余为 0

# 绘制曲线
plt.figure(figsize=(8, 6))  # 设置图形大小
plt.plot(x, y, 'b-')  # 蓝色实线
plt.fill_between(x, y, where=(x >= a) & (x <= b), color='skyblue', alpha=0.4)  # 填充区域
plt.axhline(0, color='black', linewidth=0.5)  # x 轴
# plt.axvline(a, color='m', linestyle='--')  # 下界
# plt.axvline(b, color='green', linestyle='--')  # 上界
plt.axvline(55.9, color='red', linestyle='--', label='0.795 quantile')  # 上界
plt.fill_between(x, y, where=(x >= a) & (x <= 55.9), color='skyblue')  # 填充区域


# 添加标签和标题
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Uniform Distribution PDF (40, 60)')
plt.legend()

# 设置 y 轴范围以清晰显示
plt.ylim(0, density * 1.2)

# 显示图形
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()