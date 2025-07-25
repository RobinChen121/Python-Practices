"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 2025/7/24, 20:26
@Desc    :

"""

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

# 可交互的绘图函数
def plot_sine(frequency=1.0):
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(frequency * x)
    plt.figure(figsize=(6, 3))
    plt.plot(x, y)
    plt.title(f"sin({frequency}x)")
    plt.grid(True)
    plt.show()

# 生成滑块
interact(plot_sine, frequency=FloatSlider(min=0.1, max=10.0, step=0.1, value=1.0));
