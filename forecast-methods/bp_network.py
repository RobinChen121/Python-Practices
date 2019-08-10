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


# initialize a network
def initialize_network(num_inputs, num_hidden, num_outputs):
    network = list()

    # 神经网络包括输入和输出
    hidden_layer = [{}] * num_hidden
    for i in range(num_hidden):  # 初始权重和阈值为0-1之间的随机数
        hidden_layer[i] = {'weights': [random() for i in range(num_inputs)], 'bias': random()}
    network.append(hidden_layer)

    output_layer = [{}] * num_outputs
    for i in range(num_outputs):  # 初始权重和阈值为0-1之间的随机数
        output_layer[i] = {'weights': [random() for i in range(num_hidden)], 'bias': random()}
    network.append(output_layer)

    return network


# calculate neuron activation for an input, 计算一个输入的激活值
def activate(weights, bias, inputs):
    activation = bias
    for i in range(len(weights)):
        activation += weights[i] * inputs[i]

    return activation


# transfer neuron activation, 传递激活值
# 下面使用了一个单级传递函数（losig），激活函数有很多
# 本程序输入与输出层的传递函数一样
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# forward propagate input to a network output,
# 前向传播，将输入的数据输出
def forward_propagate(network, row):
    inputs = row
    for layers in network:
        new_inputs = []
        for neuron in layers:  # 输入的数据传递进入隐含层，然后再传递出去
            activation = activate(neuron['weights'], neuron['bias'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# calculate the derivative of an neuron output
# the derivative of the transfer function
# 传递函数（sigmoid函数）的一阶导数，可以推出来
def transfer_derivative(output):
    return output * (1 - output)


# 反向计算传递的误差，并存在神经元里
def bp_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:  # 若不是输出层
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])  # 加权计算每个神经元的误差：等于加权的 delta
                errors.append(error)
        else:
            for j in range(len(layer)):  # 对输出层中的所有神经元计算误差
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]  # 计算神经元对应的 delta 值
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


seed(1)
net = initialize_network(3, 3, 1)
#for layers in net:
    #print(layers)
row1 = [1, 0, 0]
outputs = forward_propagate(net, row1)
#print(outputs)
expect = [0, 1, 1]
bp_error(net, expect)
for layers in net:
    print(layers)
