"""
@Python version : 3.12
@Author  : Zhen Chen
@Email   : chen.zhen5526@gmail.com
@Time    : 19/10/2025, 17:29
@Desc    : 

"""
import matplotlib
matplotlib.use("TkAgg")   # 或者 "Qt5Agg"，具体取决于你环境中装了哪个
import numpy as np
import matplotlib.pyplot as plt

# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成输入数据
x = np.linspace(-10, 10, 200)
y = sigmoid(x)

# 绘图
plt.figure(figsize=(7, 5))
plt.plot(x, y, label='Sigmoid Function', color='blue', linewidth=2)
plt.title('Sigmoid Activation Function', fontsize=14)
plt.xlabel(r'$x$', fontsize=12)
plt.ylabel(r'$\varphi(x)$', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()