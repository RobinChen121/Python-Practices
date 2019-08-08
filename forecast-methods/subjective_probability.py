#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/8 10:44
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description:  主观概率法，本质是估计每个可能预测值的概率，
                然后求出期望作为预测值

"""

import numpy as np


def subjective_probability(forecast_data, forecast_probability):
    forecast_num = np.dot(np.array(forecast_data), np.array(forecast_probability).T)
    return forecast_num


forecast = [112, 97, 64]
probability = [0.25, 0.5, 0.25]
forecast_value = subjective_probability(forecast, probability)
print('主观概率法的预测值为：%.2f' % forecast_value)
