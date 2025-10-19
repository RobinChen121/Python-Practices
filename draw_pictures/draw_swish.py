"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 19/10/2025, 21:22
@Desc    : 

"""
import matplotlib
matplotlib.use("TkAgg")   # 或者 "Qt5Agg"，具体取决于你环境中装了哪个
import numpy as np
import matplotlib.pyplot as plt

# 定义 swish 函数
def swish(x, alpha=1.0):
    return x / (1 + np.exp(-x))

# 生成输入数据
x = np.linspace(-5, 5, 400)
y = swish(x)

# 绘图
plt.figure()
plt.plot(x, y, label='swish', color='blue', linewidth=2)
plt.title('Swish Activation Function', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel(r'$\varphi(x)$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
