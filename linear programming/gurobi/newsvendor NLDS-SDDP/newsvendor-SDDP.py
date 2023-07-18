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

from gurobipy import *
import time
import itertools
import random

import sys 
sys.path.append("..") 
from tree import generate_sample, get_tree_strcture



ini_I = 0
vari_cost = 1
unit_back_cost = 10
unit_hold_cost = 2
mean_demands = [10, 10]
sample_nums = [10, 10]
T = len(mean_demands)
trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = 1
for i in sample_nums:
    scenario_numTotal *= i

# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    sample_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
# samples_detail = [[5, 15], [5, 15]]
scenarios_full = list(itertools.product(*sample_detail)) 


iter = 0
iter_num = 8
N = 30 # sampled number of scenarios for forward computing

theta_iniValue = 0 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = theta_iniValue*T, vtype = GRB.CONTINUOUS, name = 'theta_2')
m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)
q_value = 0
theta_value = 0

# cuts
slope1_stage = []
intercept1_stage = []
slopes = [[ [] for n in range(N)] for t in range(T-1)]
intercepts = [[ [] for n in range(N)] for t in range(T-1)]
q_values = [0 for iter in range(iter_num)]
q_sub_values = [[[0 for n in range(N)] for t in range(T-1)] for iter in range(iter_num)]
start = time.process_time()
while iter < iter_num:  
    
    # sample a numer of scenarios from the full scenario tree
    random.seed(10000)
    sample_scenarios= random.sample(scenarios_full, N) # sampling without replacement
    sample_scenarios.sort() # sort to make same numbers together
    
    # forward
    if iter > 0:
        m.addConstr(theta >= slope1_stage[-1]*q + intercept1_stage[-1])
    m.update()
    m.optimize()
    m.write('iter' + str(iter) + '_main2.lp')    
    m.write('iter' + str(iter) + '_main2.sol')
    
    q_values[iter] = q.x
    theta_value = theta.x
    z = m.objVal
    
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    I_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    # B is the quantity of lost sale
    B_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[t][n].addVar(lb = -theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]

    q_forward_values = [[0 for n in range(N)] for t in range(T)] 
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    theta_forward_values = [[0 for n in range(N)] for t in range(T)]
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]
            
            # put those cuts in the front
            if iter > 0 and t < T - 1:
                for i in range(iter):
                    for nn in range(N):
                        m_forward[t][n].addConstr(theta_forward[t][n] >= slopes[t][nn][i]*(I_forward[t][n]- B_forward[t][n] + q_forward[t][n]) + intercepts[t][nn][i])
                           
            if t == T - 1:                   
                m_forward[t][n].setObjective(unit_hold_cost*I_forward[t][n] + unit_back_cost*B_forward[t][n], GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(vari_cost*q_forward[t][n] + unit_hold_cost*I_forward[t][n] + unit_back_cost*B_forward[t][n] + theta_forward[t][n], GRB.MINIMIZE)
            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I + q_values[iter] - demand)
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] - B_forward_values[t-1][n] + q_forward_values[t][n] - demand)
            
            # optimize
            m_forward[t][n].optimize()
            # m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-2.lp')
            
            I_forward_values[t][n] = I_forward[t][n].x 
            B_forward_values[t][n] = B_forward[t][n].x      
            if t < T - 1:
                q_forward_values[t][n] = q_forward[t][n].x
                q_sub_values[iter][t][n] = q_forward[t][n].x
                theta_forward_values[t][n] = theta_forward[t][n].x
    
    # backward
    m_backward = [[[Model() for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    q_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    I_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T)]
    # B is the quantity of lost sale
    B_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward = [[[m_backward[t][n][k].addVar(lb = -theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T - 1)]

    q_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    I_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    B_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    pi_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    pi_rhs_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
    
    # it is better t in the first loop
    for t in range(T - 1, -1, -1):
       for n in range(N):
            K = len(sample_detail[t])
            for k in range(K):
                demand = sample_detail[t][k]
                
                 # put those cuts in the front
                if iter > 0 and t < T - 1:
                    for i in range(iter):
                        for nn in range(N):
                             m_backward[t][n][k].addConstr(theta_backward[t][n][k] >= slopes[t][nn][i]*(I_backward[t][n][k]- B_backward[t][n][k] + q_backward[t][n][k]) + intercepts[t][nn][i])
            
                if t == T - 1:                   
                    m_backward[t][n][k].setObjective(unit_hold_cost*I_backward[t][n][k] + unit_back_cost*B_backward[t][n][k], GRB.MINIMIZE)
                else:
                    m_backward[t][n][k].setObjective(vari_cost*q_backward[t][n][k] + unit_hold_cost*I_backward[t][n][k] + unit_back_cost*B_backward[t][n][k] + theta_backward[t][n][k], GRB.MINIMIZE)
                if t == 0:   
                    m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == ini_I + q_values[iter] - demand)
                else:
                    m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == I_forward_values[t-1][n] - B_forward_values[t-1][n] + q_forward_values[t-1][n] - demand)
                    
                # optimize
                m_backward[t][n][k].optimize()
                # if t == 0 and n == 0 and iter > 0:
                #     m_backward[t][n][k].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(k+1) +'-2back.lp')
                # if t > 0:
                #     m_backward[t][n][k].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(k+1) +'-2back.lp')
                
                pi = m_backward[t][n][k].getAttr(GRB.Attr.Pi)
                rhs = m_backward[t][n][k].getAttr(GRB.Attr.RHS)
                if t < T - 1:
                    num_con = len(pi)
                    for kk in range(num_con-1):
                        pi_rhs_values[t][n][k] += pi[kk]*rhs[kk]
                    pi_rhs_values[t][n][k] += -pi[-1]*demand 
                else:
                    pi_rhs_values[t][n][k] = -pi[-1] * demand
                pi_values[t][n][k] = pi[-1]
            
            if iter > 0 and t == 1:
                print()
            avg_pi = sum(pi_values[t][n]) / K
            avg_pi_rhs = sum(pi_rhs_values[t][n]) / K
            
            # recording cuts
            if t == 0 and n == 0:
                slope1_stage.append(avg_pi)
                intercept1_stage.append(avg_pi_rhs)
            elif t > 0:
                slopes[t-1][n].append(avg_pi)
                intercepts[t-1][n].append(avg_pi_rhs)   
            print()
            
    iter += 1

end = time.process_time()
print('********************************************')
print('final expected total costs is %.2f' % z)
print('ordering Q in the first peiod is %.2f' % q_values[iter-1])
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)





