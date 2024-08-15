#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:51:37 2024

@author: zhenchen

@disp:  transfer the multi stage model to a two stage model.
    
    
"""

from gurobipy import *
import itertools
import random
import time
import numpy as np

import sys 
import os
parent_directory = os.path.abspath('..')
sys.path.append(parent_directory) 
# sys.path.append("..") # for mac
from tree import *



demands = [[30,30,30,30,30],
[50, 46, 38, 28, 14],
[14,23,33,46,50],
[47,30,6,30,54],
[9,30,44,30,8],
[63,27,10,24,1],
[25, 46, 140, 80, 147],
[14,24,71,118,49],
[13,35,79,43,44],
[15,56,19,84,136]]

demand_pattern = 8
mean_demands1 = demands[demand_pattern - 1] # higher average demand vs lower average demand
mean_demands2 = [i*0.5 for i in mean_demands1] # higher average demand vs lower average demand

# pk1 = [0.25, 0.5, 0.25]
# pk2= pk1
# xk1 = [mean_demands1[0]-10, mean_demands1[0], mean_demands1[0]+10]
# xk2 = [mean_demands2[0]-5, mean_demands2[0], mean_demands2[0]+5]

cov1 = 0.25 # lower variance vs higher variance
cov2 = 0.5
sigmas1 = [cov1*i for i in mean_demands1]
sigmas2 = [cov2*i for i in mean_demands2]
T = len(mean_demands1)

ini_Is = [0, 0]
ini_cash = 0
vari_costs = [1, 2]
prices = [5, 10] # lower margin vs higher margin
MM = len(prices)
unit_salvages = [0.5* vari_costs[m] for m in range(MM)]
overhead_cost = [100 for t in range(T)]

r0 = 0  # when it is 0.01, can largely slow the compuational speed
r1 = 0.1
r2 = 2 # penalty interest rate for overdraft exceeding the limit, does not affect computation time
U = 500 # overdraft limit

sample_num = 5 # change 1

# detailed samples in each period
trunQuantile = 0.9999 # affective to the final ordering quantity
sample_details1 = [[0 for i in range(sample_num)] for t in range(T)]
sample_details2 = [[0 for i in range(sample_num)] for t in range(T)]
for t in range(T):
    # sample_details1[t] = generate_samples_gamma(sample_num, trunQuantile, mean_demands1[t], betas[0])
    # sample_details2[t] = generate_samples_gamma(sample_num, trunQuantile, mean_demands2[t], betas[1])
    # sample_details1[t] = generate_samples(sample_num, trunQuantile, mean_demands1[t])
    # sample_details2[t] = generate_samples(sample_num, trunQuantile, mean_demands2[t])
    sample_details1[t] = generate_samples_normal(sample_num, trunQuantile, mean_demands1[t], sigmas1[t])
    sample_details2[t] = generate_samples_normal(sample_num, trunQuantile, mean_demands2[t], sigmas2[t])
    # sample_details1[t] = generate_samples_discrete(sample_num, xk1, pk1)
    # sample_details2[t] = generate_samples_discrete(sample_num, xk2, pk2)

# sample_details1 = [[10, 30], [10, 30], [10, 30]] # change 2
# sample_details2 = [[5, 15], [5, 15], [5, 15]]

theta_iniValue = -1000 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
q2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_2')
W0 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^0')
W1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^1')
W2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^2')
theta = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_2')

m.setObjective(overhead_cost[0] + vari_costs[0]*q1 + vari_costs[1]*q2 + r2*W2 + r1*W1 - r0*W0 + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))
m.addConstr(W1 <= U)
m.addConstr(-vari_costs[0]*q1 - vari_costs[1]*q2- W0 + W1 + W2 == overhead_cost[0] - ini_cash)


# cuts recording arrays
iter_limit = 500
time_limit = 3600
N = 5 # sampled number of scenarios in forward computing, change 3
slope_stage1_1 = []
slope_stage1_2 = []
slope_stage1_3 = []
intercept_stage1 = []
slopes1 = []
slopes2 = []
slopes3 = []
intercepts = []
q1_values = [] 
qpre1_values = [] 
q2_values = [] 
qpre2_values = [] 
W0_values = []
W1_values = []
W2_values = []

iter = 0
time_pass = 0
stop_condition = 'iter_limit'
start = time.process_time()

while iter < iter_limit and time_pass < time_limit: # and means satify either one will exist the loop
    
    
    pass