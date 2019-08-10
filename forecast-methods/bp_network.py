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
    for i in range(num_outputs): # 初始权重和阈值为0-1之间的随机数
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
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))





seed(1)
net = initialize_network(3, 3, 1)
for layer in net:
    print(layer)