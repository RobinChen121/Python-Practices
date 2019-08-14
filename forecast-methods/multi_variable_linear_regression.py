#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/8 16:05
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description: forecasting by multi variable linear regression, generally need over 10 data
#                 通过多元线性回归进行预测，一般需要 10个 以上的数据

"""

import numpy as np
import statsmodels.api as sm
import pandas as pd
import os


def linear_regression_by_formula(history_x_data, history_y_data):
    """
    输入自变量和因变量数据
    :param history_x_data: 自变量数据，类型为 ndarray
    :param history_y_data:  因变量数据，类型 为 ndarray
    :return: 预测值，斜率，截距
    """
    T = len(history_y_data)
    X = np.c_[np.ones((T, 1)), history_x_data]
    Y = history_y_data
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    beta_0 = beta[0]
    beta_1 = beta[1:len(beta)]
    forecast_datas = [0] * T
    for i in range(T):
        forecast_datas[i] = np.dot(np.array(history_x_data)[i, :], np.array(beta_1)) + beta_0
    PE = [0] * T  # absolute percentage error
    for i in range(T):
        PE[i] = abs(forecast_datas[i] - history_y_data[i]) / history_y_data[i]
    MPE = sum(PE) / T
    return beta, forecast_datas


def linear_regression_by_package(history_x_data, history_y_data):
    """
    通过调用 statesmodels 包进行回归分析

    :param history_x_data: 自变量数据，类型为 ndarray
    :param history_y_data:  因变量数据，类型 为 ndarray
    :return: 预测值，斜率，截距
    """

    T = len(history_y_data)
    X = np.array(history_x_data)
    Y = history_y_data
    X = sm.add_constant(X)
    result = sm.OLS(Y, X).fit()

    beta = result.params
    forecast_data = result.predict(X)
    return beta, forecast_data, result.summary()


x = [[227.9, 397.6], [270.4, 423.8], [285.9, 462.6], [340.7, 544.9], [397.7, 601.5], [433.9, 686.3], [482.2, 708.6],
     [564.9, 784], [644.4, 921.6], [884, 1221]]
y = [[46871], [41361], [39017], [65645], [28569], [29649], [29679], [33439], [42145], [50864]]
betas, forecast_values = linear_regression_by_formula(x, y)
print('多元线性方程的截距和斜率分别为：')
print(betas)

betas, __, summary = linear_regression_by_package(x, y)
print('预测包的回归结果：')
print(summary)

# arr = pd.DataFrame(np.c_[y, x])  # test for output to excel files
# arr.to_excel('history data.xlsx')
