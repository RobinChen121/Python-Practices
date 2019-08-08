#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/8 10:21
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description: delphi forecasting. 德尔菲法，选择一些专业人士预测，取不同专家预测值的中位数。
               上四分位与下四分位 <= 1.1 表示意见相对收敛

"""

import numpy as np


def delphi(expert_data):
    forecast_data = np.median(expert_data)
    upper_q = np.quantile(expert_data, 0.75)
    lower_q = np.quantile(expert_data, 0.25)
    return forecast_data, lower_q, upper_q


expert_values = [28, 29, 24, 25, 27]
forecast_value, __, __ = delphi(expert_values)
print('德尔菲法的预测值为： %.2f' % forecast_value)