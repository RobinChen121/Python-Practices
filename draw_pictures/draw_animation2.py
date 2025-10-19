"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 20/09/2025, 16:13
@Desc    : 

"""
import matplotlib
matplotlib.use("TkAgg")   # 或者 "Qt5Agg"，具体取决于你环境中装了哪个
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 数据长度
frames = 100

# 创建画布和坐标轴
fig, ax = plt.subplots()
ax.set_xlim(0, frames)
ax.set_ylim(-1.5, 1.5)

# 初始化曲线对象
line, = ax.plot([], [])

# 初始化函数：给曲线设置初始空数据
def init():
    line.set_data([], [])
    return line,

# 更新函数：每帧更新 y 数据
def update(frame):
    x = np.arange(frame)
    y = np.sin(0.1 * x)
    line.set_data(x, y)
    return line,

# 创建动画对象
ani = FuncAnimation(
    fig, update, frames=frames, init_func=init, blit=True, interval=50
)

# 保存为 GIF，需要 pillow
ani.save('animation.gif', writer='pillow')

# 可选：显示动画窗口
plt.show()
