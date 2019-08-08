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
import matplotlib.pyplot as plt
import statsmodels.api as sm


def linear_regression_by_formula(history_x_data, history_y_data):
    T = len(history_y_data)
    X = np.c_(np.ones(T, 1), history_x_data)
    return
