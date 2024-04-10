#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:18:39 2024

@author: zhenchen

@disp:  business overdraft for lead time in 2 product problem;

using random gamma generator seems not very well
"""

from gurobipy import *
import itertools
import random
import time
import numpy as np

import sys 
sys.path.append("..") 
from tree import generate_gamma_sample



def generate_samples(sample_details, sample_num, N):
    MM = len(sample_details)
    T = len(sample_details[0])
    random_samples = [[[0 for t in range(T)] for n in range(N)] for m in range(MM)]
    for m in range(MM):        
        for n in range(N):
            for t in range(T):
                random_index = np.random.randint(low = 0, high = sample_num-1)
                random_samples[m][n][t] = sample_details[m][t][random_index]        
    return random_samples
    


T = 3
ini_Is = [0, 0]
ini_cash = 0
vari_costs = [1, 2]
prices = [5, 10] # lower margin vs higher margin
MM = len(prices)
unit_salvage = [0.5* vari_costs[m] for m in range(MM)]
overhead_cost = [100 for t in range(T)]

r0 = 0.01
r1 = 0.1
r2 = 3 # penalty interest rate for overdraft exceeding the limit
U = 2000 # overdraft limit

sample_num = 10 # sample number in one stage when forming the scenario tree
scenario_numTotal = sample_num ** T

# gamma distribution:mean demand is shape / beta and variance is shape / beta^2
# beta = 1 / scale
# shape = demand * beta
# variance = demand / beta
mean_demands =[30, 10] # higher average demand vs lower average demand
betas = [10, 1] # lower variance vs higher variance

# detailed samples in each period
trunQuantile = 0.9999 # affective to the final ordering quantity
sample_details = [[[0 for i in range(sample_num)] for t in range(T)] for m in range(MM)]
for m in range(MM):
    for t in range(T):
            sample_details[m][t] = generate_gamma_sample(sample_num, trunQuantile, mean_demands[m], betas[m])


theta_iniValue = -5000 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
q2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_2')
W0 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^0')
W1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^1')
W2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^2')
theta = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_2')

m.setObjective(vari_costs[0]*q1 + vari_costs[1]*q2 + r2*W2 + r1*W1 - r0*W0 + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))
m.addConstr(W1 <= U)
m.addConstr(-vari_costs[0]*q1 - vari_costs[1]*q2- W0 + W1 + W2 == overhead_cost[0] - ini_cash)

# cuts recording arrays
iter_num = 4
N = 10 # sampled number of scenarios in forward computing
slope_stage1 = []
intercept_stage1 = []
slopes1 = [[[ 0 for n in range(N)] for t in range(T)] for i in range(iter_num)]
slopes2 = [[[0 for n in range(N)] for t in range(T)] for i in range(iter_num)]
slopes3 = [[[0 for n in range(N)] for t in range(T)] for i in range(iter_num)]
intercepts = [[[0 for n in range(N)] for t in range(T-1)] for i in range(iter_num)]
q_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
qpre_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
W0_values = [0 for iter in range(iter_num)]
W1_values = [0 for iter in range(iter_num)]
W2_values = [0 for iter in range(iter_num)]

start = time.process_time()
iter = 0
while iter < iter_num:  
    sample_scenarios = generate_samples(sample_details, sample_num, N)
    pass
    sample_scenarios[0].sort() # sort to make same numbers together
    sample_scenarios[1].sort()