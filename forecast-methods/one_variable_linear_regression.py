#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/8 11:24
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description:  forecasting by single variable linear regression, generally need over 10 data
                通过一元线性回归进行预测，一般需要 10个 以上的数据

"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.interactive(True)  # 防止调试不能出现交互式界面

def linear_regression_by_formula(history_data):
    """
    通过公式计算线性回归参数

    :param history_data:历史数据
    :return: 预测值， 斜率， 截距
    """
    lxx = 0
    lxy = 0
    T = len(history_data)

    x = range(1, T + 1)
    y = history_data
    x_mean = np.mean(x)
    y_mean = np.mean(history_data)

    for i in range(T):
        lxx = lxx + (x[i] - x_mean) ** 2
        lxy = lxy + (x[i] - x_mean) * (y[i] - y_mean)

    beta_1 = lxy / lxx
    beta_0 = y_mean - beta_1 * x_mean

    forecast_data_all = [0] * (T + 1)
    for i in range(1, T + 1):
        forecast_data_all[i] = x[i - 1] * beta_1 + beta_0
    forecast_data_all[T] = (x[T - 1] + 1) * beta_1 + beta_0
    return forecast_data_all[T], beta_0, beta_1, forecast_data_all


def linear_regression_by_package(history_data):
    """
    通过调用 statesmodels 包进行回归分析

    :param history_data: 历史数据
    :return: 回归包生成的各种结果
    """
    T = len(history_data)
    x = np.arange(1, T + 1)
    y = history_data

    x = sm.add_constant(x)
    result = sm.OLS(y, x).fit()
    beta_0, beta_1 = result.params
    forecast_data = result.predict(x)
    forecast_next_data = beta_0 + beta_1 * (T + 1)
    arr = np.append(forecast_data, forecast_next_data)
    PE = [0] * T  # absolute percentage error
    for i in range(T):
        PE[i] = abs(forecast_data[i] - history_data[i]) / history_data[i]
    MPE = sum(PE) / T
    return forecast_next_data, MPE, beta_0, beta_1, result.summary(), arr


def draw_picture(history_data, forecast_data):
    # 解决 plt 中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    T = len(history_data)
    plt.plot(range(1, T + 1), history_data, '-o', label='历史数据')
    plt.plot(range(1, T + 2), forecast_data, '-o', label='预测数据')
    plt.legend()
    plt.title('线性回归方法')
    plt.xticks(range(1, T + 2))
    plt.grid(axis='y')
    plt.show()
    return


history_value = [2231.3, 2288.9, 2402.7, 2540.5, 2948.6, 3126.1, 3543.6, 3954.1, 4156.9, 4162.2, 4490.2, 4834.5, 4778.6,
                 5061.5]
forecast_value, a, b, __ = linear_regression_by_formula(history_value)
print('预测值为： %.2f' % forecast_value)
print('预测的方程为： y = %.2fx + %.2f' % (a, b))

forecast_value, error, a, b, summary, forecast_values = linear_regression_by_package(history_value)
print('通过统计包的预测值为： %.2f' % forecast_value)
print('通过统计包的预测方程为： y = %.2fx + %.2f' % (a, b))
print('通过统计包的预测平均绝对差百分比为： %.2f%%' % (error * 100))
print('预测包的回归结果：')
print(summary)
draw_picture(history_value, forecast_values)
