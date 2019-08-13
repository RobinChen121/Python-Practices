#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/11 16:44
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description: 该方法混合了加权平均方法，结合数据波动规律，与需求经验，进行预测

"""


def rule1(arr):
    forecast_value = 0
    return forecast_value


def rule2(arr):
    forecast_value = arr.mean()
    return forecast_value


def rule3(arr):
    column_num = len(arr)
    if arr[- 1] < 50000:
        forecast_value = arr[- 1] / 2
        return
    if arr[column_num - 2] > 8 * arr[column_num - 1] and arr[column_num - 1] > 0.1:
        forecast_value = arr[column_num - 1] / 5
        return

    #    if arr[column_num - 1] > 10 * arr[column_num - 2] and arr[column_num - 2] > 0.1:
    #        forecast_value = arr[column_num - 1]
    #        forcast_rule[i] = 8
    #        return
    # 若上一年小于5万，则按上一年数据处理

    if (max(arr) - min(arr)) / min(arr) > 3:  # 剔除极端值
        if (max(arr)) / sum(arr) > 0.5:
            forecast_value = (sum(arr) - max(arr)) / (len(arr) - 1)
        if (min(arr)) / sum(arr) < 0.1:
            forecast_value = (sum(arr) - min(arr)) / (len(arr) - 1)
        if (max(arr)) / sum(arr) > 0.5 and (min(arr)) / sum(arr) < 0.1:
            forecast_value = (sum(arr) - min(arr) - max(arr)) / (len(arr) - 2)
    else:
        forecast_value = arr.mean()
    return forecast_value


def rule4(arr):
    forecast_value = arr[-1]
    return forecast_value


def hybrid_forecast(history_data):
    column_num = len(history_data)
    if history_data[0] in ['布电线', '钢芯铝绞线、钢绞线', '低压(刀)开关']:
        forecast_value = history_data[column_num - 1] / 3

    if history_data[0] in ['布电线', '10千伏交流避雷器', '10千伏跌落式熔断器', '配变计量箱']:
        forecast_value = history_data[column_num - 1]

    # 若连续两年下降 10 倍，则 19 年也下降10倍
    if min(history_data[column_num - 3: column_num]) > 1 and history_data[column_num - 3] > \
            10 * history_data[column_num - 2] and history_data[column_num - 2] > 10 * history_data[column_num - 1]:
        forecast_value = history_data[column_num - 1] / 10

    # 若 17, 18 年为零，则 19 年也为零                        
    if history_data[column_num - 1] < 0.1 and history_data[column_num - 2] < 0.1:
        forecast_value = rule1(history_data)

        # 如果从14年开始，连续出现两个零以上，之后的数据都不为零，则对不为零的数据移动平均
    for j in range(1, column_num):
        if sum(history_data[0: j]) < 10 and min(history_data[j: column_num]) > 10:
            forecast_value = rule2(history_data[j: column_num])
            break
    # 如果出现 5 年均有数据，但波动幅度（最大值-最小值)/最小值 超过3倍
    # 我们按最近 3 年的平均数做预测；如果未超过5倍，则取所有年度的平均值
    # 若预测年度前一年比预测前第二年下降或上涨超过10倍，则取前一年的值
    if min(history_data[:]) > 10:
        forecast_value = rule3(history_data[:])

    # 如果只有 17 年缺失，18 年出现极端值，则按 18 年的数据处理；
    if history_data[column_num - 2] == 0 and min(history_data[0: column_num - 2]) > 10 \
            and history_data[column_num - 1] > 3 * history_data[0: column_num - 2].mean():
        forecast_value = rule4(history_data)

    # 如果只有 14 年为 0，则将 14 年数据剔除，计算规则与 3 相同
    if history_data[0] == 0 and min(history_data[1: column_num]) > 10:
        forecast_value = rule3(history_data[1: column_num])

    # 若零交替出现，则按交替规律预测
    #    if (sum(history_data[range(0, column_num, 2)]) < 10 and min(history_data[range(1, column_num, 2)]) > 10) or \
    #        (sum(history_data[range(1, column_num, 2)]) < 10 and min(history_data[range(0, column_num, 2)]) > 10):
    #            forecast_value = history_data[column_num - 2]
    #            forcast_rule[i] = 6
    #            
    # 若 18 年不为零，14 年不为零，而其他年存在零，则将零剔除，取剩下值的移动平均
    if history_data[column_num - 1] > 10 and history_data[0] > 10 \
            and min(history_data[0: column_num - 1]) < 10:
        arr = history_data[:]
        arr = arr[arr > 10]
        forecast_value = arr.mean()

    # 若只有前两年有数据，则取平均值
    if min(history_data[column_num - 1], history_data[column_num - 2]) > 0.1 and \
            sum(history_data[1: column_num - 2]) < 0.1:
        arr = history_data[:]
        arr = arr[arr > 0.1]
        if arr[-2] > 2.5 * arr[-1]:
            forecast_value = arr[-1] / 1.5
        else:
            forecast_value = arr.mean()
