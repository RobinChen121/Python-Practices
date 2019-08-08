# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:49:42 2019

@author: zhen chen

Python version: 3.7

Email: 15011074486@163.com

Description: python codes for one-level exponential smoothing forecasting method, 一次指数平滑的 python 代码，
             本方法的初始预测值取第一个历史数据
    
"""

import matplotlib.pyplot as plt
import numpy as np


# one level exponential smoothing method
# 一次指数平滑
# for stationary data, data number should better over than 5
# 适用于波动性不大的平稳数据，数据量最好大于5个

def one_level_es(history_data):
    """

    :param history_data: 历史数据
    :return: 最优平滑系数，未来一期的预测值，所有预测值，平均绝对百分比误差
    """

    T = len(history_data)
    best_MPE = 1  # default MPE,默认最优平滑系数对应的百分比误差
    best_alpha = 0  # default alpha,默认最优平滑系数
    for alpha in np.arange(0, 1, 0.1):
        forecast_data = [0] * (T + 1)
        PE = [0] * T  # absolute percentage error
        forecast_data[0] = history_data[0]
        for i in range(1, T + 1):
            forecast_data[i] = alpha * history_data[i - 1] + (1 - alpha) * forecast_data[i - 1]
        for i in range(T):
            PE[i] = abs(forecast_data[i] - history_data[i]) / history_data[i]
        MPE = sum(PE) / T
        if MPE < best_MPE:
            best_MPE = MPE
            best_alpha = alpha

    return best_alpha, forecast_data[T], forecast_data, best_MPE


def draw_picture(history_data, forecast_data):
    # 解决 plt 中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    T = len(history_data)
    plt.plot(range(1, T + 1), history_data, '-o', label='历史数据')
    plt.plot(range(1, T + 2), forecast_data, '-o', label='预测数据')
    plt.legend()
    plt.title('一次指数平滑方法')
    plt.xticks(range(1, T + 2))
    plt.grid(axis='y')
    plt.show()
    return


demand_data = [100.4, 100.7, 99.2, 101.2, 103.9, 101.8, 101.5, 104.8, 105.9, 99.3, 103.3, 105.4, 102.6, 102.6]
coe, next_forecast_data, all_forecast_data, forecast_error = one_level_es(demand_data)
print('一次指数平滑————')
print('未来一期的预测值: %.2f' % next_forecast_data)
print('平均百分比预测误差为: %.2f%%' % (forecast_error * 100))
print('最优平滑系数为: %.1f' % coe)
draw_picture(demand_data, all_forecast_data)
