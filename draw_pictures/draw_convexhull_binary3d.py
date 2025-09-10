"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 10/09/2025, 15:03
@Desc    : 

"""
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

# 1. 三维二进制点
binary_points = np.array(list(product([0, 1], repeat=3)))  # shape (8,3)

# 2. 任意点 x
x = np.array([0.3, 0.7, 0.5])

# 3. 创建 3D 图形
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

# 绘制二进制点
ax.scatter(binary_points[:,0], binary_points[:,1], binary_points[:,2],
           c='blue', s=100, label='Binary points')

# 绘制单位立方体的边
cube_edges = [
    [(0,0,0),(1,0,0)], [(0,0,0),(0,1,0)], [(0,0,0),(0,0,1)],
    [(1,1,1),(0,1,1)], [(1,1,1),(1,0,1)], [(1,1,1),(1,1,0)],
    [(1,0,0),(1,1,0)], [(1,0,0),(1,0,1)],
    [(0,1,0),(1,1,0)], [(0,1,0),(0,1,1)],
    [(0,0,1),(1,0,1)], [(0,0,1),(0,1,1)]
]

for edge in cube_edges:
    p1, p2 = edge
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k--', alpha=0.5)

# 绘制点 x
ax.scatter(x[0], x[1], x[2], c='red', s=100, label='Point x')

# 绘制 x 到每个顶点的连线示意凸组合
for bp in binary_points:
    ax.plot([x[0], bp[0]], [x[1], bp[1]], [x[2], bp[2]], 'gray', linestyle=':', alpha=0.3)

# 设置图形
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_zlim(0,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Convex Hull of Binary Vectors in 3D')
ax.legend()
plt.show()
