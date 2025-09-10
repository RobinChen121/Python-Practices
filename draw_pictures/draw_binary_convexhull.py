"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 10/09/2025, 22:08
@Desc    : 

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# 定义 f 在 {0,1} 上的值
f0, f1 = 1.0, 3.0

# x 范围
xs = np.linspace(0, 1, 100)
# 下包曲线
h = (1 - xs) * f0 + xs * f1

# 作图
fig, ax = plt.subplots(figsize=(7,6))

# 1. 画集合 S（竖直射线）：(0,t>=f0), (1,t>=f1)
ax.plot([0,0], [f0, max(f0,f1)+2], color="gray", linestyle="--", label="S at x=0")
ax.plot([1,1], [f1, max(f0,f1)+2], color="gray", linestyle="--", label="S at x=1")

# 2. 画凸包 C 的下边界：就是 h(x)
ax.plot(xs, h, color="blue", linewidth=2, label="lower envelope h(x)")

# 3. 画凸包 C 区域（h(x) 上方区域）
# 为了填充，取 [0,1] 区间的下边界 h，再延长到一个大高度 ymax
ymax = max(f0,f1)+2
polygon_x = np.concatenate([[0], xs, [1]])
polygon_y = np.concatenate([[ymax], h, [ymax]])
ax.fill_between(polygon_x, polygon_y, ymax, color="blue", alpha=0.2, label="C (convex hull of S)")

# 4. 标记二值点
ax.scatter([0,1], [f0,f1], color="red", zorder=5)
ax.text(0, f0-0.2, f"f(0)={f0}", color="red", ha="center")
ax.text(1, f1+0.1, f"f(1)={f1}", color="red", ha="center")

ax.set_xlim(-0.2,1.2)
ax.set_ylim(0,ymax+0.5)
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_title("Epigraph convex hull C and convex lower envelope h(x)")
ax.legend()
ax.grid(True)
plt.show()
