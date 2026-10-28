"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 26/10/2026, 12:56
@Desc:

"""

import pyreadr
import numpy as np
import pandas as pd
import os

# 获取当前脚本的目录
current_dir = os.path.dirname(os.getcwd())

# 构建数据文件的完整路径
data_file = os.path.join(current_dir, "data", "carparts.rda")

result = pyreadr.read_r(data_file)
df = next(iter(result.values()))

# 将数据转换为数值类型，非数值的将被转换为 NaN
# error='coerce' 参数会将无法转换为数值的值设置为 NaN
# 默认 error='ignore' 参数会忽略无法转换为数值的值，保留原始值，
df = df.apply(pd.to_numeric, errors="coerce")

# 删除存在缺失值的产品列
df = df.dropna(axis=1)  # axis=1 表示按列删除，axis=0 表示按行删除，默认值为 axis=0

# 计算每种产品的均值和方差
means = df.mean(
    axis=0
)  # axis=0 表示按列计算均值，axis=1 表示按行计算均值， 默认值为 axis=0
# 得到 Pandas Series 和 NumPy 一维数组没有行、列之分，只有一个轴。只有二维数据才有明确的行和列
variances = df.var(
    axis=0, ddof=1
)  # ddof=1 表示使用样本方差，ddof=0 表示使用总体方差，默认值为 1

# 离散比率
dispersion_ratio = variances / means

# 过度离散的产品比例
overdispersed = (dispersion_ratio.dropna() > 1).mean()

print("Complete series:", df.shape[1])
print("Overdispersed:", overdispersed)
print("Average dispersion ratio:", dispersion_ratio.mean())


# Average gap


import numpy as np


def average_gap(x):
    indices = np.flatnonzero(x > 0)

    if len(indices) < 2:
        return np.nan

    return np.diff(indices).mean()


# 方法一：每个产品的 Average Gap 再平均
gaps = df.apply(average_gap, axis=0)

print("Method 1:", gaps.mean())


# 方法二：将所有产品的间隔合并后取平均，这个得到的结果和那本预测教科书上的结果一样
all_gaps = []

for col in df.columns:
    indices = np.flatnonzero(df[col].to_numpy() > 0)
    all_gaps.extend(np.diff(indices))

print("Method 2:", np.mean(all_gaps))


# 方法三：Croston 常用的 ADI = T / 正需求次数
positive_counts = (df > 0).sum(axis=0)

adi = df.shape[0] / positive_counts.replace(0, np.nan)

print("Method 3:", adi.mean())


# 方法四：所有产品的总观测期数 / 总正需求次数
pooled_adi = df.size / (df > 0).to_numpy().sum()

print("Method 4:", pooled_adi)

# draw picture for the average demand per month
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(df.mean(axis=1), marker="o", linestyle="-", color="blue")
plt.title("Average Demand per Month")
plt.xlabel("Product")
plt.ylabel("Average Demand")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
