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
sample_nums = [10, 10]
T = len(mean_demands)
trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = reduce(lambda x, y: x * y, sample_nums, 1)

# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    sample_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
# samples_detail = [[5, 15], [5, 15]]
scenarios_full = list(itertools.product(*sample_detail)) 


iter = 1
iter_num = 15
N = 20 # sampled number of scenarios for forward computing

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
    sample_scenarios= random.sample(scenarios_full, N) # sampling without replacement
    sample_scenarios.sort() # sort to make same numbers together
    
    # forward
    m.update()
    m.optimize()
    q_value = q.x
    theta_value = theta.x
    z = m.objVal
    
    m_forward = [[Model() for t in range(T)] for n in range(N)] 
    q_forward = [[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for t in range(T - 1)] for n in range(N)]
    I_forward = [[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for t in range(T)] for n in range(N)]
    # B is the quantity of lost sale
    B_forward = [[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for t in range(T)] for n in range(N)]
    theta_forward = [[m_forward[n][t].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for t in range(T - 1)] for n in range(N)]

    q_forward_values = [[0 for t in range(T)] for n in range(N)]  
    I_forward_values = [[0 for t in range(T)] for n in range(N)]
    B_forward_values = [[0 for t in range(T)] for n in range(N)]
    theta_forward_values = [[0 for t in range(T)] for n in range(N)]
    
    for n in range(N):
        for t in range(T):
            demand = sample_scenarios[n][t]
            if t == 0:   
                if T > 1:
                    m_forward[n][t].setObjective(vari_cost*q_forward[n][t] - price*(demand - B_forward[n][t]) + theta_forward[n][t], GRB.MINIMIZE)
                    m_forward[n][t].addConstr(theta_forward[n][t] >= theta_iniValue*(T-1-t))
                else:
                    m_forward[n][t].setObjective(-price*(demand - B_forward[n][t]), GRB.MINIMIZE)
                m_forward[n][t].addConstr(I_forward[n][t] - B_forward[n][t] == ini_I + q_value - demand)
                print('')               
            else:
                if t == T - 1:                   
                    m_forward[n][t].setObjective(-price*(demand - B_forward[n][t]), GRB.MINIMIZE)
                else:
                    m_forward[n][t].setObjective(vari_cost*q_forward[n][t] - price*(demand - B_forward[n][t]) + theta_forward[n][t], GRB.MINIMIZE)
                    m_forward[n][t].addConstr(theta_forward[n][t] >= theta_iniValue*(T-1-t))
                m_forward[n][t].addConstr(I_forward[n][t] - B_forward[n][t] == I_forward_values[n][t-1] + q_forward_values[n][t] - demand)
                
            # optimize
            m_forward[n][t].optimize()
            I_forward_values[n][t] = I_forward[n][t].x 
            B_forward_values[n][t] = B_forward[n][t].x      
            if t < T - 1:
                q_forward_values[n][t] = q_forward[n][t].x
                theta_forward_values[n][t] = theta_forward[n][t].x
    
    # backward
    m_backward = [[[Model() for t in range(T)] for n in range(N)] for k in range(sample_nums[t])] 
    q_backward = [[[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for t in range(T - 1)] for n in range(N)]for k in range(sample_nums[t])]
    I_backward = [[[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for t in range(T)] for n in range(N)]for k in range(sample_nums[t])]
    # B is the quantity of lost sale
    B_backward = [[[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for t in range(T)] for n in range(N)]for k in range(sample_nums[t])]
    theta_backward = [[[m_forward[n][t].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for t in range(T - 1)] for n in range(N)]for k in range(sample_nums[t])]

    q_backward_values = [[[0 for t in range(T)] for n in range(N)] for k in range(sample_nums[t])]
    I_backward_values = [[[0 for t in range(T)] for n in range(N)] for k in range(sample_nums[t])]
    B_backward_values = [[[0 for t in range(T)] for n in range(N)] for k in range(sample_nums[t])]
    theta_backward_values = [[[0 for t in range(T)] for n in range(N)] for k in range(sample_nums[t])]
    
    for n in range(N):
        for t in range(T-1, -1, -1):
            K = len(sample_detail[t])
            slope = [0 for k in range(K)]
            intercept = [0 for k in range(K)]
            for k in range(K):
                demand = sample_scenarios[t][k]
                if t == 0:   
                    if T > 1:
                        m_backward[k][n][t].setObjective(vari_cost*q_backward[k][n][t] - price*(demand - B_backward[k][n][t]) + theta_backward[k][n][t], GRB.MINIMIZE)
                        m_backward[k][n][t].addConstr(theta_backward[k][n][t] >= theta_iniValue*(T-1-t))
                    else:
                        m_backward[k][n][t].setObjective(-price*(demand - B_backward[k][n][t]), GRB.MINIMIZE)
                    m_backward[k][n][t].addConstr(I_backward[k][n][t] - B_backward[k][n][t] == ini_I + q_value - demand)
                    print('')               
                else:
                    if t == T - 1:                   
                        m_backward[k][n][t].setObjective(-price*(demand - B_backward[k][n][t]), GRB.MINIMIZE)
                    else:
                        m_backward[k][n][t].setObjective(vari_cost*q_backward[k][n][t] - price*(demand - B_backward[k][n][t]) + theta_backward[k][n][t], GRB.MINIMIZE)
                        m_backward[k][n][t].addConstr(theta_backward[k][n][t] >= theta_iniValue*(T-1-t))
                    m_backward[k][n][t].addConstr(I_backward[k][n][t] - B_backward[k][n][t] == I_forward_values[n][t-1] + q_forward_values[k][n][t] - demand)
                    
                # optimize
                m_backward[k][n][t].optimize()
                pi = m_backward[k][n][t].getAttr(GRB.Attr.Pi)
                avg_pi = sum(pi) / K
            
            print()





