#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/8 23:33
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description: 灰色系统预测方法，
               本程序为 GM(1, 1)模型，若有多个自变量，则可以构建 GM(1, h)模型，
               更复杂的模型还有 GM(n, h) 模型

"""

import numpy as np
import math
import matplotlib.pyplot as plt


def gm(history_data):
    """
    灰色模型 GM(1, 1)

    :param history_data: 历史数据
    :return: 预测值，预测序列，误差
    """
    T = len(history_data)
    sum_data = [0] * T
    sum_data[0] = history_data[0]
    for i in range(1, T):
        sum_data[i] = sum_data[i - 1] + history_data[i]
    B = np.zeros([T - 1, 2])
    for i in range(T - 1):
        B[i, 0] = -0.5 * (sum_data[i] + sum_data[i + 1])
        B[i, 1] = 1
    X = history_data[1:T]

    [alpha, u] = (np.dot(np.linalg.inv(np.dot(B.T, B)), np.dot(B.T, X))).T

    forecast_data = [0] * (T + 1)
    forecast_data[0] = history_data[0]
    for i in range(1, T + 1):
        forecast_data[i] = (forecast_data[0] - u / alpha) * math.exp(-alpha * i) + u / alpha

    forecast_data_final = [0] * (T + 1)
    forecast_data_final[0] = forecast_data[0]
    for i in range(1, T + 1):
        forecast_data_final[i] = forecast_data[i] - forecast_data[i - 1]
    PE = [0] * T  # absolute percentage error
    for i in range(T):
        PE[i] = abs(forecast_data_final[i] - history_data[i]) / history_data[i]
    MPE = sum(PE) / T

    return forecast_data_final[T], forecast_data, MPE


def draw_picture(history_data, forecast_data):
    # 解决 plt 中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    T = len(history_data)
    plt.plot(range(1, T + 1), history_data, '-o', label='历史数据')
    plt.plot(range(1, T + 2), forecast_data, '-o', label='预测数据')
    plt.legend()
    plt.title('灰色模型预测')
    plt.xticks(range(1, T + 2))
    plt.grid(axis='y')
    plt.show()
    return


history_value = [394, 7269, 3954, 1723]
forecast_value_final, forecast_values, error = gm(history_value)
print('预测值为： %.2f' % forecast_value_final)
print('预测误差为： %.2f%%' % (error * 100))
draw_picture(history_value, forecast_values)
