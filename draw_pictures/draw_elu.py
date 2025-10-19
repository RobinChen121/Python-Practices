"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 19/10/2025, 18:24
@Desc    : 

"""
import matplotlib
matplotlib.use("TkAgg")   # 或者 "Qt5Agg"，具体取决于你环境中装了哪个
import numpy as np
import matplotlib.pyplot as plt

# 定义 ELU 函数
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 生成输入数据
x = np.linspace(-5, 5, 400)
y1 = elu(x, alpha=1.0)
y2 = elu(x, alpha=0.5)

# 绘图
plt.figure()
plt.plot(x, y1, label='ELU α=1.0', color='blue', linewidth=2)
plt.plot(x, y2, label='ELU α=0.5', color='orange', linestyle='--', linewidth=2)
plt.title('ELU Activation Function', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel(r'$\varphi(x)$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
