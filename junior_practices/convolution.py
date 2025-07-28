"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 27/07/2025, 12:11
@Desc    : 

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import simps


# 定义函数
def f(x):
    return np.exp(-x ** 2)


def g(x):
    return np.exp(-(x - 2) ** 2)


x = np.linspace(-5, 5, 400)
t_vals = np.linspace(-3, 7, 100)

fx = f(x)
gx = g(x)  # 原始g(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-5, 10)
ax.set_ylim(-0.1, 1.3)
ax.set_xlabel('x / t')
ax.set_title('Convolution of two continuous functions')

line_f, = ax.plot(x, fx, 'g-', lw=2, label='f(x)')
line_g_fixed, = ax.plot(x, gx, 'orange', lw=2, label='g(x) (fixed)')
line_g_slide, = ax.plot([], [], 'r-', lw=2, label='g(t - x)')
line_conv, = ax.plot([], [], 'b-', lw=2, label='(f * g)(t)')

fill = ax.fill_between(x, 0, 0, color='b', alpha=0.3)
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

ax.legend(loc='upper right')

conv_x, conv_y = [], []


def init():
    global conv_x, conv_y, fill
    conv_x, conv_y = [], []  # 清空旧数据
    line_conv.set_data([], [])
    line_g_slide.set_data([], [])
    text.set_text('')
    if fill is not None:
        fill.remove()
        fill = ax.fill_between(x, 0, 0, color='b', alpha=0.3)  # 重设为透明
    return line_conv, line_g_slide, fill, text


def update(i):
    global fill, conv_x, conv_y
    t = t_vals[i]

    gx_slide = g(t - x)
    prod = fx * gx_slide
    conv_val = simps(prod, x)

    line_g_slide.set_data(x, gx_slide)

    if fill is not None:
        fill.remove()
    fill = ax.fill_between(x, 0, prod, color='b', alpha=0.3)

    conv_x.append(t)
    conv_y.append(conv_val)
    line_conv.set_data(conv_x, conv_y)

    text.set_text(f't = {t:.2f}, convolution = {conv_val:.4f}')

    return line_g_slide, fill, line_conv, text


import matplotlib.patches as mpatches

# 之前已有的图例句柄
handles, labels = ax.get_legend_handles_labels()

# 新增一个 Patch，代表填充区域
fill_patch = mpatches.Patch(color='lightblue', alpha=0.5, label=r'Product area: $f(x) \cdot g(t - x)$')

# 添加到图例句柄和标签中
handles.append(fill_patch)
labels.append(r'Product area: $f(x) \cdot g(t - x)$')

ax.legend(handles, labels, loc='upper right')
ax.set_xlabel('x (function variable) and t (sliding parameter)')

ani = FuncAnimation(fig, update, frames=len(t_vals), interval=100, blit=True, repeat=True, init_func=init)
# ani.save('convolution.gif', dpi=72)
plt.show()
