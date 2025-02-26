"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/23 13:27
Description: 
    

"""

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
#
# # 定义正态分布参数
# mu, sigma = 8, 3
#
# # 每个阶段的数据点数
# points_per_stage = 100
#
# # 定义局部坐标（每个阶段的 x 轴范围），这里取 [-4, 4]，便于展示完整的正态分布
# x_local = np.linspace(mu - 4 * sigma, mu + 4 * sigma, points_per_stage)
# pdf_values = norm.pdf(x_local, mu, sigma)
#
# # 定义三个阶段的时间偏移量
# offsets = [0, 20, 40]
#
# plt.figure(figsize=(10, 6))
#
# for i, offset in enumerate(offsets):
#     # 对每个阶段，在局部 x 轴基础上添加时间偏移
#     x_stage = x_local + offset
#     plt.plot(x_stage, pdf_values, label=f"stage {i+1}")
#
# plt.xlabel("stage number")
# plt.ylabel("probability density function")
# plt.legend()
# plt.title("3 stage normal distribution")
# plt.legend()
# plt.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 定义三个阶段的时间范围和均值
time_stages = [100, 100, 100]  # 每个阶段100个时间点
mean_demands = [40, 60, 80]  # 每个阶段的均值
std_dev = 5  # 标准差，控制波动幅度

# 生成平稳需求数据
y1 = (np.random.normal(mean_demands[0], std_dev, time_stages[0]))
y2 = (np.random.normal(mean_demands[1], std_dev, time_stages[1]))
y3 = (np.random.normal(mean_demands[2], std_dev, time_stages[2]))

# 绘制图像
plt.figure(figsize=(10, 6))
x1 = np.arange(0, time_stages[0])
x2 = np.arange(time_stages[0], time_stages[0] + time_stages[1])
x3 = np.arange(
    time_stages[0] + time_stages[1],
    time_stages[0] + time_stages[1] + time_stages[2],
)
plt.plot(x1, y1, color="r", label=f"Stage 1 Mean: {mean_demands[0]}")
plt.plot(x2, y2, color="g", label=f"Stage 2 Mean: {mean_demands[1]}")
plt.plot(x3, y3, color="m", label=f"Stage 3 Mean: {mean_demands[2]}")

# plt.axhline(y=mean_demands[0], color='r', linestyle='--', label=f"Stage 1 Mean: {mean_demands[0]}")
# plt.axhline(y=mean_demands[1], color='g', linestyle='--', label=f"Stage 2 Mean: {mean_demands[1]}")
# plt.axhline(y=mean_demands[2], color='m', linestyle='--', label=f"Stage 3 Mean: {mean_demands[2]}")

plt.title("Three Stages of Stationary Demand")
plt.xlabel("Time")
plt.ylabel("Demand")
plt.legend()
plt.grid(True)
plt.show()
