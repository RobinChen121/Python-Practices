"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 10/09/2025, 14:59
@Desc    : draw the convex hull of binary vectors

"""
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# 1. 二维二进制点
binary_points = np.array(list(product([0, 1], repeat=2)))  # shape (4,2)

# 2. 任意点 x
x = np.array([0.3, 0.7])

# 3. 绘图
fig, ax = plt.subplots(figsize=(6,6))

# 绘制二进制点
ax.scatter(binary_points[:,0], binary_points[:,1], c='blue', s=100, label='Binary points')

# 绘制凸包（单位正方形）
square = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
ax.plot(square[:,0], square[:,1], 'k--', label=r'Convex hull $[0,1]^2$')

# 绘制点 x
ax.scatter(x[0], x[1], c='red', s=100, label='Point x')

# 绘制 x 到每个顶点的连线示意凸组合
for bp in binary_points:
    ax.plot([x[0], bp[0]], [x[1], bp[1]], 'gray', linestyle=':', alpha=0.5)

# 设置图形
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.1, 1.1)
ax.set_xticks([0,0.25,0.5,0.75,1])
ax.set_yticks([0,0.25,0.5,0.75,1])
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
ax.set_title('Convex Hull of Binary Vectors in 2D')
plt.show()
