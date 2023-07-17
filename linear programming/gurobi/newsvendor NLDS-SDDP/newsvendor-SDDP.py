#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 08:44:23 2023

@author: zhenchen

@disp:  
    
    
"""


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
    
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    I_forward = [[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    # B is the quantity of lost sale
    B_forward = [[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[n][t].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]

    q_forward_values = [[0 for n in range(N)] for t in range(T)] 
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    theta_forward_values = [[0 for n in range(N)] for t in range(T)]
    
    for t in range(T):
        for t in range(T):
            demand = sample_scenarios[n][t]
            if t == 0:   
                if T > 1:
                    m_forward[t][n].setObjective(vari_cost*q_forward[t][n] - price*(demand - B_forward[t][n]) + theta_forward[t][n], GRB.MINIMIZE)
                    m_forward[t][n].addConstr(theta_forward[t][n] >= theta_iniValue*(T-1-t))
                else:
                    m_forward[t][n].setObjective(-price*(demand - B_forward[t][n]), GRB.MINIMIZE)
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I + q_value - demand)
                print('')               
            else:
                if t == T - 1:                   
                    m_forward[t][n].setObjective(-price*(demand - B_forward[t][n]), GRB.MINIMIZE)
                else:
                    m_forward[t][n].setObjective(vari_cost*q_forward[t][n] - price*(demand - B_forward[t][n]) + theta_forward[t][n], GRB.MINIMIZE)
                    m_forward[t][n].addConstr(theta_forward[t][n] >= theta_iniValue*(T-1-t))
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] + q_forward_values[t][n] - demand)
                
            # optimize
            m_forward[n][t].optimize()
            I_forward_values[t][n] = I_forward[t][n].x 
            B_forward_values[t][n] = B_forward[t][n].x      
            if t < T - 1:
                q_forward_values[t][n] = q_forward[t][n].x
                theta_forward_values[t][n] = theta_forward[t][n].x
    
    # backward
    m_backward = [[[Model()  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    q_backward = [[[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    I_backward = [[[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T)]
    # B is the quantity of lost sale
    B_backward = [[[m_forward[n][t].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward = [[[m_forward[n][t].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T - 1)]

    q_backward_values = [[[0  for k in range(sample_nums[t]) for n in range(N)]] for t in range(T)]
    I_backward_values = [[[0  for k in range(sample_nums[t]) for n in range(N)]] for t in range(T)]
    B_backward_values = [[[0  for k in range(sample_nums[t]) for n in range(N)]] for t in range(T)]
    theta_backward_values = [[[0  for k in range(sample_nums[t]) for n in range(N)]] for t in range(T)]
    pi_values = [[[0  for k in range(sample_nums[t]) for n in range(N)]] for t in range(T)]
    pi_rhs_values = [[[0  for k in range(sample_nums[t]) for n in range(N)]] for t in range(T)] 
    
    for n in range(N):
        for t in range(T - 1, -1, -1):
            K = len(sample_detail[t])
            slope = [0 for k in range(K)]
            intercept = [0 for k in range(K)]
            for k in range(K):
                demand = sample_detail[t][k]
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
                rhs = m_backward[k][n][t].getAttr(GRB.Attr.RHS)
                if t < T - 1:
                    num_con = len(pi)
                    for kk in range(num_con-1):
                        pi_rhs_values[k][n][t] += pi[kk]*rhs[kk]
                    pi_rhs_values[k][n][t] += -pi[-1]*demand 
                else:
                    pi_rhs_values[k][n][t] = -pi[-1] * demand
                pi_values[t][j] = pi[-1]
            avg_pi = sum(pi__values[t]) / t_nodeNum[t]
            sum_pi_rhs = 0
            for j in range(t_nodeNum[t]): 
                sum_pi_rhs += pi_rhs_values[t][j]
            avg_pi_rhs = sum_pi_rhs / t_nodeNum[t]
            
            print()





