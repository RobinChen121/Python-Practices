# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:49:42 2019

@author: zhen chen

Python version: 3.7

Email: robinchen@swu.edu.cn

Description: python codes for one-level exponential smoothing forecasting method, 一次指数平滑的 python 代码
    
"""

import matplotlib.pyplot as plt


# one level exponential smoothing method
# 一次指数平滑
# for stationary data, data number should better over than 5
# 适用于波动性不大的平稳数据，数据量最好大于5个

def one_level_es(history_data, alpha):
    T = len(history_data)
    forecast_data = [0] * (T + 1)
    PE = [0] * T  # absolute percentage error
    forecast_data[0] = history_data[0]
    for i in range(1, T + 1):
        forecast_data[i] = alpha * history_data[i - 1] + (1 - alpha) * forecast_data[i - 1]
    for i in range(T):
        PE[i] = abs(forecast_data[i] - history_data[i]) / history_data[i]

    MPE = sum(PE) / T
    return forecast_data[T], forecast_data, MPE


def draw_picture(history_data, forecast_data):
    # 解决 plt 中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    T = len(history_data)
    plt.plot(range(1, T + 1), history_data, '-o', label='历史数据')
    plt.plot(range(1, T + 2), forecast_data, '-o', label='预测数据')
    plt.legend()
    plt.xticks(range(1, T + 2))
    plt.grid(axis='y')
    plt.show()
    return


demand_data = [100.4, 100.7, 99.2, 101.2, 103.9, 101.8, 101.5, 104.8, 105.9, 99.3, 103.3, 105.4, 102.6, 102.6]
alpha = 0.3
next_forecast_data, all_forecast_data, forecast_error = one_level_es(demand_data, alpha)
print('The next forecast value is: %.2f' % next_forecast_data)
print('The mean absolute percentage error in this forecasting is: %.2f' % forecast_error)
draw_picture(demand_data, all_forecast_data)
