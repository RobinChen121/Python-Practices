#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:52:47 2023

@author: zhenchen

@disp:  
    
    
"""

from sample_tree import generate_sample, get_tree_strcture, getSizeOfNestedList
from gurobipy import *
import time
from functools import reduce
import itertools
import random
import time


start = time.process_time()
ini_I = 0
vari_cost = 1
price = 10
unit_back_cost = 0
unit_hold_cost = 0
mean_demands = [10, 10]
sample_nums = [10, 10, 10, 10]
T = len(mean_demands)
trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = reduce(lambda x, y: x * y, sample_nums, 1)

# detailed samples in each period
samples_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    samples_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
# samples_detail = [[5, 15], [5, 15]]
scenarios_full = list(itertools.product(*samples_detail)) 


iter = 1
iter_num = 15
sample_num = 20

theta_iniValue = -300 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = theta_iniValue*T, vtype = GRB.CONTINUOUS, name = 'theta_2')
m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)
q_value = 0
theta_value = 0


while iter <= iter_num:  
    
    # sample a numer of scenarios from the full scenario tree
    sample_scenarios= random.sample(scenarios_full, sample_num) # sampling without replacement
    sample_scenarios.sort() # sort to make same numbers together
    node_values, node_index = get_tree_strcture(sample_scenarios)
    
    # forward
    m.update()
    m.optimize()
    q_value = q.x
    theta_value = theta.x
    z = m.objVal
    
    forward_m = [[Model() for t in range(T)] for n in range(sample_num)] 
    
    for n in range(sample_num):
        
        print(n)
    
    # backward





