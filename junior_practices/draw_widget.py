"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 2025/7/25, 11:26
@Desc    : 

"""
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 或 'Qt5Agg'，取决于系统支持

import mpl_interactions.ipyplot as iplt

x = np.linspace(0, np.pi, 100)
tau = np.linspace(1, 10, 10)
beta = np.linspace(0.001, 1, 10)


def f(x, tau, beta):
    return np.sin(x * tau) * x**beta


fig, ax = plt.subplots()
controls = iplt.plot(x, f, tau=tau, beta=beta, slider_formats={"beta": "{:.3e}"})
plt.show()