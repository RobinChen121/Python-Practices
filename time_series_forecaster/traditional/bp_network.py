#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

# @Time    : 2019/8/9 22:22
# @Author  : Zhen Chen
# @Email   : 15011074486@163.com

# Python version: 3.7
# Description: forecasting by back propagation network,
              利用反向传播的神经网络预测。
              一般需要上千组数据训练用，100组左右数据预测用

"""

from random import random
from random import seed
from math import exp
import numpy as np


# initialize a network
def initialize_network(num_inputs, num_hidden, num_outputs):
    network = list()

    # 神经网络包括输入层、隐含层和输出层
    # 下面分别生成从输入层到隐含层，以及从隐含层到输出层的初始权重
    # 初始权重为0-1之间的随机数
    hidden_layer = [{}] * num_hidden
    for i in range(num_hidden):
        hidden_layer[i] = {'weights': [random() for i in range(num_inputs + 1)]}  # weights 数组中最后一个元素是阈值
    network.append(hidden_layer)

    output_layer = [{}] * num_outputs
    for i in range(num_outputs):
        output_layer[i] = {'weights': [random() for i in range(num_hidden + 1)]}  # weights 数组中最后一个元素是阈值
    network.append(output_layer)

    return network


# calculate neuron activation for an input and weights, 计算一个输入的激活值
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):  # 最后一个权重值是阈值，不用加
        activation += weights[i] * inputs[i]

    return activation


# transfer neuron activation, 传递函数
# 下面使用了一个单级传递函数（losig），传递函数有很多
# 本程序输入与输出层的传递函数一样
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# forward propagate input to a network output,
# 前向传播，将输入层的数据输出到所有隐含层和输出层的数据中
# input_values_row 表示输入层的数据
def forward_propagate(network, input_values_row):
    inputs = input_values_row
    for layer in network:
        layer_output_values = []
        for neuron in layer:  # 输入的数据传递进入隐含层，然后再传递出去
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            layer_output_values.append(neuron['output'])
        inputs = layer_output_values
    return inputs  # 返回输出层的输出值


# calculate the derivative of an neuron output
# the derivative of the transfer function
# 传递函数（sigmoid函数）的一阶导数，可以推出来
def transfer_derivative(output):
    return output * (1 - output)


# 反向计算传递的误差，并存在每个神经元的 delta 里
# expect_output_values 表示输出层的所有期望输出值
def compute_deltas(network, expect_output_values_row):
    for i in reversed(range(len(network))):  # 反向计算
        layer = network[i]
        errors = list()  # 一层里面所有神经元的误差在一个 list 里
        if i != len(network) - 1:  # 若不是输出层
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:  # i + 1 表示下一层
                    error += (neuron['weights'][j] * neuron['delta'])  # 加权计算每个神经元的误差：等于加权的 delta
                errors.append(error)
        else:  # 若在输出层
            for j in range(len(layer)):  # 对输出层中的所有神经元计算误差
                neuron = layer[j]
                errors.append(expect_output_values_row[j] - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]  # 计算每个神经元对应的 delta 值
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# update network weights with error
# 更新权重（在网络训练时用）
# 更新的原理利用了梯度下降
# 每组输入的数值，都更新一次权重
def update_weights(network, input_values_row, learning_rate):
    for i in range(len(network)):
        inputs = input_values_row[: -1]  # 因为 input_values 中最后一个元素是输出值
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]  # 本网络的输入是上一个神经网络的输出
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]  # 更新权重
            neuron['weights'][-1] += learning_rate * neuron['delta']  # 更新阈值, 阈值是最后一个权重


# train a network for a fixed number of epochs
# 以一定次数训练网络
def train_net(network, train_input_values, train_output_values, learning_rate, epochs):
    if len(train_output_values.shape) == 1:
        num_outputs = 1
    else:
        num_outputs = train_output_values.shape[1]
    for epoch in range(epochs):
        sum_error = 0
        for input_values_row in train_input_values:
            output = forward_propagate(network, input_values_row)
            expected_output_values_row = [train_output_values[-i] for i in range(num_outputs)]
            # expected = [0 for i in range(n_outputs)]
            # expected[row[-1]] = 1
            # 每一组各输出的误差叠加
            sum_error += sum([(expected_output_values_row[i]-output[i])**2 for i in range(len(expected_output_values_row))])
            # 所有输入组的输出误差叠加，结果为 sum_error
            compute_deltas(network, expected_output_values_row)
            update_weights(network, input_values_row, learning_rate)
        print('>epoch = %d, learning rate = %.3f, error = %.3f' % (epoch, learning_rate, sum_error))


# Make a prediction with a network
# 通过训练后的网络做预测
def predict(network, test_input_values):
    output_values = forward_propagate(network, test_input_values)
    return output_values


# 归一化，将一个矩阵每一列归一化处理
def preminmax(arr):
    row_num = len(arr)
    column_num = len(arr[0])
    # out_arr = [[0] * column_num] * row_num  # 这种方式会造成 2D list每一行的引用完全相同
    # out_arr = [[0] * column_num for __ in range(row_num)]
    out_arr = np.zeros((row_num, column_num))  # 直接用 np array 类型避免上面的情况
    arr = np.array(arr)
    column_max_ps = []
    column_min_ps = []
    for i in range(column_num):
        column_max = max(arr[:, i])  # 因为 arr 是 list 类型，若 arr 是 numpy 类型，则可以按 matlab 的方式引用
        column_min = min(arr[:, i])
        column_max_ps.append(column_max)
        column_min_ps.append(column_min)
        for j in range(row_num):
            out_arr[j][i] = 2 * (arr[j][i] - column_min) / (column_max - column_min) - 1
    return out_arr, column_max_ps, column_min_ps


seed(1)
dataset = [[970062, 18718.3, 103922, 5560.1, 8300.1, 6955.81],
           [985793, 21826.2, 104844, 7225.8, 9415.6, 9810.4],
           [1045899, 26937.3, 107256, 9119.6, 10993.7, 12443.12],
           [1115902, 35260, 111059, 11271, 14270.4, 14410.22],
           [1180396, 48108.5, 118729, 20381.9, 18622.9, 17042.94],
           [1234938, 59810.5, 129034, 23499.9, 23613.8, 20019.3],
           [1298421, 70142.5, 133032, 24133.8, 28360.2, 22974],
           [1278218, 78060.9, 133460, 26967.2, 31252.9, 24941.1],
           [1267427, 83024.3, 129834, 26849.7, 33378.1, 28406.2],
           [1293008, 88479.2, 131935, 29896.2, 35647.9, 29854.7],
           [1358682, 98000.5, 135048, 39273.2, 39105.7, 32917.7],
           [1401786, 108068.2, 143875, 42183.6, 43055.4, 37213.5],
           [1483447, 119095.7, 150656, 51378.2, 48135.9, 43499.9],
           [1564492, 134977, 171906, 70483.5, 52516.3, 55566.6],
           [1706412, 159453.6, 196648, 95539.1, 59501, 70477.43],
           [1862066, 183617.4, 216219, 116921.8, 67176.6, 88773.61],
           [2037060, 215904.4, 232167, 140974, 76410, 109998.16],
           [2275822, 266422, 242279, 166863.7, 89210, 137323.94],
           [2585973, 316030.3, 260552, 179921.5, 108487.7, 172828.4],
           [2825222, 340320, 274619, 150648.1, 132678.4, 224598.77],
           [3241807, 399759.5, 296916, 201722.1, 156998.4, 278121.85],
           [3696961, 472115, 317987, 236402, 183918.6, 311485.13]]

data_array = np.array(dataset)
normal_data, max_ps, min_ps = preminmax(data_array)
input_data = normal_data[0: 17, 1: len(normal_data[0])]
output_data = normal_data[0: 17, 1]  # 输出的数据在第一列
print(normal_data)
n_outputs = 1
n_inputs = len(input_data[0]) - n_outputs
n_hidden = 1
net = initialize_network(n_inputs, n_hidden, n_outputs)
# for layer in net:
#     print(layer)
learn_rate = 0.3
num_epoch = 100
train_net(net, input_data, output_data, learn_rate, num_epoch)
out_values = predict(net, [5, 10, 10, 10, 10])
print(out_values)
# for row in dataset:
#     prediction = predict(net, row)
#     print('Expected = %d' % row[-1])
#     print('forecast:')
#     print(list(prediction))
