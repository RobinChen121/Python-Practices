#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/9 14:33
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description: holt 指数模型预测，该方法适合有趋势，但没季节性的数据，
               一般数据量要 5 个以上
               分别编写了靠公式，以及靠统计包两种计算方式。
               两种方式计算结果不一定一致，初始值、初始趋势的选取对结果有影响
               靠公式的方式，通过枚举不同系数，比 spss 的运行效果好。
               靠统计包的方式，必须给定两个系数

"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import Holt


def holt_es(history_data):
    T = len(history_data)

    forecast_data = [0] * (T + 1)
    trend = [0] * T
    smooth_data = [0] * T

    best_MPE = 1  # default MPE,默认最优平滑系数对应的百分比误差
    best_alpha = 0  # default alpha,默认最优平滑系数
    best_beta = 0  # default beta, 默认最优趋势系数
    for alpha in np.arange(0.1, 1, 0.01):
        for beta in np.arange(0.1, 1, 0.01):
            smooth_data[0] = history_data[0] - 0.01
            forecast_data[0] = history_data[0]
            trend[0] = history_data[1] - history_data[0]
            PE = [0] * T  # absolute percentage error

            for i in range(1, T):
                smooth_data[i] = alpha * history_data[i] + (1 - alpha) * (smooth_data[i - 1] + trend[i - 1])
                trend[i] = beta * (smooth_data[i] - smooth_data[i - 1]) + (1 - beta) * trend[i - 1]
                forecast_data[i] = smooth_data[i - 1] + trend[i - 1]
            forecast_data[T] = smooth_data[T - 1] + trend[T - 1]
            for i in range(T):
                PE[i] = abs(forecast_data[i] - history_data[i]) / history_data[i]
            MPE = sum(PE) / T
            if MPE < best_MPE:
                best_MPE = MPE
                best_alpha = alpha
                best_beta = beta

    return forecast_data[T], best_alpha, best_beta, best_MPE, forecast_data


def holt_package(history_data):
    result = Holt(history_data).fit(smoothing_level=0.93, smoothing_slope=0.35, optimized=False)
    T = len(history_data)
    forecast_data = [0] * T
    PE = [0] * T

    forecast_data = result.fittedvalues
    forecast_data_last = result.fcastvalues
    forecast_data = np.append(forecast_data, forecast_data_last)
    for i in range(T):
        PE[i] = abs(forecast_data[i] - history_data[i]) / history_data[i]
    MPE = sum(PE) / T
    print(result.summary)
    return forecast_data_last, forecast_data, MPE


def draw_picture(history_data, forecast_data):
    # 解决 plt 中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    T = len(history_data)
    plt.plot(range(1, T + 1), history_data, '-o', label='历史数据')
    plt.plot(range(1, T + 2), forecast_data, '-o', label='预测数据')
    plt.legend()
    plt.title('Holt 指数平滑方法')
    plt.xticks(range(1, T + 2))
    plt.grid(axis='y')
    plt.show()
    return


demand_data = [2231.3, 2288.9, 2402.7, 2540.5, 2948.6, 3126.1, 3543.6, 3954.1, 4156.9, 4162.2, 4490.2, 4834.5]
forecast_value_last, coe1, coe2, error, forecast_values = holt_es(demand_data)
print('holt 指数平滑————')
print('未来一期的预测值: %.2f' % forecast_value_last)
print('平均百分比预测误差为: %.2f%%' % (error * 100))
print('最优平滑系数为: %.2f, 最优趋势系数为：%.2f' % (coe1, coe2))
#  print(forecast_values)
draw_picture(demand_data, forecast_values)

forecast_value_last, forecast_values, error = holt_package(demand_data)
print('holt 指数平滑(通过软件包）————')
print('未来一期的预测值: %.2f' % forecast_value_last)
print('平均百分比预测误差为: %.2f%%' % (error * 100))
draw_picture(demand_data, forecast_values)
