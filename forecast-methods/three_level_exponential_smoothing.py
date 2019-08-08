#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/8 10:11
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description: three-level exponential smoothing for forecasting, it is a quadratic function
               of time t. 三次指数平滑方法，它是一个关于时间 t 的二次函数

"""

import matplotlib.pyplot as plt
import numpy as np


def three_level_es(history_data):
    """

    :param history_data: 历史数据
    :return: 最优平滑系数，未来一期的预测值，所有预测值，平均绝对百分比误差
    """

    T = len(history_data)
    best_MPE = 1  # default MPE, 默认最优平滑系数对应的百分比误差
    best_alpha = 0  # default alpha, 默认最优平滑系数
    for alpha in np.arange(0, 1, 0.1):
        forecast_data_1 = [0] * (T + 1)  # 一次指数平滑的预测结果
        forecast_data_2 = [0] * (T + 1)  # 二次指数平滑的预测结果，比一次指数平滑少一个值
        forecast_data_3 = [0] * (T + 1)  # 三次指数平滑的预测结果
        PE = [0] * T  # absolute percentage error
        forecast_data_1[0] = history_data[0]
        for i in range(1, T + 1):
            forecast_data_1[i] = alpha * history_data[i - 1] + (1 - alpha) * forecast_data_1[i - 1]
            forecast_data_2[0] = forecast_data_1[1]
            forecast_data_3[0] = forecast_data_1[1]
            if i > 0:
                forecast_data_2[i] = alpha * forecast_data_1[i] + (1 - alpha) * forecast_data_2[i - 1]
                forecast_data_3[i] = alpha * forecast_data_2[i] + (1 - alpha) * forecast_data_3[i - 1]
        for i in range(T):
            PE[i] = abs(forecast_data_3[i] - history_data[i]) / history_data[i]
        MPE = sum(PE) / T
        if MPE < best_MPE:
            best_MPE = MPE
            best_alpha = alpha

    return best_alpha, forecast_data_3[T - 1], forecast_data_3[1:T + 1], best_MPE


def draw_picture(history_data, forecast_data):
    # 解决 plt 中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    T = len(history_data)
    plt.plot(range(1, T + 1), history_data, '-o', label='历史数据')
    plt.plot(range(2, T + 2), forecast_data, '-o', label='预测数据')
    plt.legend()
    plt.title('三次指数平滑方法')
    plt.xticks(range(1, T + 2))
    plt.grid(axis='y')
    plt.show()
    return


demand_data = [100.4, 100.7, 99.2, 101.2, 103.9, 101.8, 101.5, 104.8, 105.9, 99.3, 103.3, 105.4, 102.6, 102.6]
coe, next_forecast_data, all_forecast_data, forecast_error = three_level_es(demand_data)
print('三次指数平滑————')
print('未来一期的预测值: %.2f' % next_forecast_data)
print('平均百分比预测误差为: %.2f%%' % (forecast_error * 100))
print('最优平滑系数为: %.1f' % coe)
draw_picture(demand_data, all_forecast_data)