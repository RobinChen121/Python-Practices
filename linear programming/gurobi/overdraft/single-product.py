#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 20:04:23 2023

@author: zhenchen

@disp:  
    overdraft for single product situation
    
"""

from gurobipy import *
import time
import itertools
import random
from functools import reduce # 从工具包 functools 中导入 reduce

import sys 
sys.path.append("..") 
from tree import generate_sample, get_tree_strcture, generate_scenario_samples


ini_I = 0
ini_cash = 0
vari_cost = 1
price = 5
mean_demands = [10, 20]
overhead_cost = [50, 50]
r0 = 0.01
r1 = 0.1
limit = 80 # overdraft limit
T = len(mean_demands)

sample_nums = [10 for t in range(T)]
scenario_numTotal = reduce(lambda x1, x2: x1*x2, sample_nums)


# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
trunQuantile = 0.9999 # can influence the final ordering quantity
for t in range(T):
    sample_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])

theta_iniValue = -500 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = theta_iniValue*T, vtype = GRB.CONTINUOUS, name = 'theta_2')
W0 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^0')
W1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^1')
m.setObjective(overhead_cost[0] + vari_cost*q + r1*W1 - r0*W0 + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))
# m.addConstr(-vari_cost*q - r1*W1 + r0*W0 >= overhead_cost[0] - ini_cash - limit)
m.addConstr(vari_cost*q - W0 + W1 == overhead_cost[0] - ini_cash)
q_value = 0

iter = 0
iter_num = 7
N = 20 # sampled number of scenarios for forward computing
while iter <= iter_num:  
    # sample a numer of scenarios from the full scenario tree
    # random.seed(10000)
    sample_scenarios = generate_scenario_samples(N, trunQuantile, mean_demands)
    sample_scenarios.sort() # sort to make same numbers together
    
    m.optimize()    
    m.write('iter' + str(iter) + '_main.lp')    
    m.write('iter' + str(iter) + '_main.sol')
    
    # forward
    
    
    iter += 1
    pass


