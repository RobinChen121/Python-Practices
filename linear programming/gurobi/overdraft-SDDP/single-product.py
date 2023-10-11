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
import numpy as np
from functools import reduce # 从工具包 functools 中导入 reduce

import sys 
sys.path.append("..") 
from tree import generate_sample, get_tree_strcture, generate_scenario_samples


ini_I = 0
ini_cash = 0
vari_cost = 1
unit_sal = 0.5
price = 5
mean_demands = [10, 10]
overhead_cost = [50, 50]
r0 = 0.01
r1 = 0
r2 = 0.1
r3 = 1
V = 25
U = 80 # overdraft limit
T = len(mean_demands)

sample_nums = [10 for t in range(T)]
scenario_numTotal = reduce(lambda x1, x2: x1*x2, sample_nums)


# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
trunQuantile = 0.9999 # can influence the final ordering quantity
for t in range(T):
    sample_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
sample_nums = [2, 2]
sample_detail = [[5, 15], [5, 15]]

theta_iniValue = -500 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = theta_iniValue*T, vtype = GRB.CONTINUOUS, name = 'theta_2')
W0 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^0')
W1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^1')
W2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^2')
W3 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^3')
m.setObjective(overhead_cost[0] + vari_cost*q + r3*W3 + r2*W2 + r1*W1 - r0*W0 + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))
m.addConstr(W1 <= V)
m.addConstr(W1 + W2 <= U)
m.addConstr(-vari_cost*q - W0 + W1 + W2 + W3 == overhead_cost[0] - ini_cash)


iter = 0
iter_num = 7
N = 4 # sampled number of scenarios for forward computing
q_values = [0 for iter in range(iter_num)]
q_sub_values = [[[0 for n in range(N)] for t in range(T-1)] for iter in range(iter_num)]
W0_values = [0 for iter in range(iter_num)]
W1_values = [0 for iter in range(iter_num)]
W2_values = [0 for iter in range(iter_num)]
W3_values = [0 for iter in range(iter_num)]

slopes1 = [[ [] for n in range(N)] for t in range(T)]
slopes2 = [[ [] for n in range(N)] for t in range(T)]
intercept = [[ [] for n in range(N)] for t in range(T)]

start = time.process_time()
while iter < iter_num:  
    # sample a numer of scenarios from the full scenario tree
    # random.seed(10000)
    sample_scenarios = generate_scenario_samples(N, trunQuantile, mean_demands)
    sample_scenarios = [[5, 5], [5, 15], [15, 5], [15, 15]]
    sample_scenarios.sort() # sort to make same numbers together
    
    # cuts
    if iter > 0:
        m.addConstr(theta >= slopes1[0][0][-1]*q + slopes2[0][0][-1]*(-vari_cost*q-r3*W3-r2*W2-r1*W1+r0*W0)\
                                + intercept[0][0][-1])
    m.update()
    
    m.optimize()    
    m.write('iter' + str(iter) + '_main.lp')    
    m.write('iter' + str(iter) + '_main.sol')
    
    q_values[iter] = q.x
    W0_values[iter] = W0.x
    W1_values[iter] = W1.x
    W2_values[iter] = W2.x
    W3_values[iter] = W3.x
    z = m.objVal
    
    # forward
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W0_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W1_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W2_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W3_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W3_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    I_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    # B is the quantity of lost sale
    B_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    C_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    
    q_forward_values = [[0 for n in range(N)] for t in range(T-1)]  
    W0_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    W1_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    W2_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    W3_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    C_forward_values = [[0 for n in range(N)] for t in range(T)]
    theta_forward_values = [[0 for n in range(N)] for t in range(T)]
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]
            
            # put those cuts in the front
            if iter > 0 and t < T - 1:
                for i in range(iter):
                    for nn in range(N): 
                        m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[t][nn][i]*(I_forward[t][n]+q_forward[t][n])\
                                    + slopes2[t][nn][i]*(-vari_cost*q_forward[t][n]-r3*W3_forward[t][n]-r2*W2_forward[t][n]\
                                    - r1*W1_forward[t][n]+r0*W0_forward[t][n]) + intercept[t][nn][i])
            if t == T - 1:                   
                m_forward[t][n].setObjective(-price*(demand - B_forward[t][n]) - unit_sal*I_forward[t][n], GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(overhead_cost[t] - price*(demand - B_forward[t][n]) + overhead_cost[t+1] + vari_cost*q_forward[t][n] + r3*W3_forward[t][n] + r2*W2_forward[t][n] + r1*W1_forward[t][n] - r0*W0_forward[t][n] + theta_forward[t][n], GRB.MINIMIZE) # 
            
            # constraints
            if t < T - 1:
                m_forward[t][n].addConstr(theta_forward[t][n] >= theta_iniValue*T)
                m_forward[t][n].addConstr(W1_forward[t][n] <= V)
                m_forward[t][n].addConstr(W1_forward[t][n] + W2_forward[t][n] <= U)
                m_forward[t][n].addConstr(C_forward[t][n] - vari_cost*q_forward[t][n] - W0_forward[t][n] + W1_forward[t][n] + W2_forward[t][n] + W3_forward[t][n] == overhead_cost[t])
                
            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I + q_values[iter] - demand)
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] + q_forward_values[t-1][n] - demand)
            if t == 0:
                m_forward[t][n].addConstr(C_forward[t][n] + price*B_forward[t][n] == ini_cash - overhead_cost[t] - vari_cost*q_values[iter]\
                                          -r1*W1_values[iter]-r2*W2_values[iter]-r3*W3_values[iter] + r0*W0_values[iter] + price*demand)
            else:
                m_forward[t][n].addConstr(C_forward[t][n] + price*B_forward[t][n] == C_forward_values[t-1][n]- overhead_cost[t] - vari_cost*q_forward_values[t-1][n] -r1*W1_forward_values[t-1][n]\
                                          -r2*W2_forward_values[t-1][n] -r3*W3_forward_values[t-1][n]+ r0*W0_forward_values[t-1][n] + price*demand) 
           
            
            # optimize
            m_forward[t][n].optimize()
            # if iter == 1 and t == 0 and n == 0:
            #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t) + '^' + str(n) + '.lp')
            #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t) + '^' + str(n) + '.sol')
            #     pass
            
            I_forward_values[t][n] = I_forward[t][n].x 
            B_forward_values[t][n] = B_forward[t][n].x    
            C_forward_values[t][n] = C_forward[t][n].x
            if t < T - 1:
                W1_forward_values[t][n] = W1_forward[t][n].x
                W2_forward_values[t][n] = W2_forward[t][n].x
                W3_forward_values[t][n] = W3_forward[t][n].x
            if t < T - 1:
                q_forward_values[t][n] = q_forward[t][n].x
                q_sub_values[iter][t][n] = q_forward[t][n].x
                theta_forward_values[t][n] = theta_forward[t][n]
            
    # backward
    m_backward = [[[Model() for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    q_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    I_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T)]
    # B is the quantity of lost sale
    B_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward = [[[m_backward[t][n][k].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T - 1)]
    C_backward = [[[m_backward[t][n][k].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    W0_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    W1_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    W2_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    W3_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'W3_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    
    pi_values = [[[[0, 0]  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    pi_rhs_value = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
    for t in range(T - 1, -1, -1):
       for n in range(N):
            K = len(sample_detail[t])
            for k in range(K):
                demand = sample_detail[t][k]
                
                # put those cuts in the front
                if iter > 0 and t < T - 1:
                    for i in range(iter):
                        for nn in range(N): # N
                             m_backward[t][n][k].addConstr(theta_backward[t][n][k] >= slopes1[t][nn][i]*(I_backward[t][n][k]+q_backward[t][n][k])\
                                                           + slopes2[t][nn][i]*(-vari_cost*q_backward[t][n][k]\
                                                     -r3*W3_backward[t][n][k]-r2*W2_backward[t][n][k]-r1*W1_backward[t][n][k]\
                                                     +r0*W0_backward[t][n][k])+ intercept[t][nn][i])
                       
                if t == T - 1:                   
                    m_backward[t][n][k].setObjective(-price*(demand - B_backward[t][n][k]) - unit_sal*I_backward[t][n][k], GRB.MINIMIZE)
                else:
                    m_backward[t][n][k].setObjective(overhead_cost[t] - price*(demand - B_backward[t][n][k]) + overhead_cost[t+1] + vari_cost*q_backward[t][n][k] + r3*W3_backward[t][n][k] + r2*W2_backward[t][n][k] + r1*W1_backward[t][n][k] - r0*W0_backward[t][n][k] + theta_backward[t][n][k], GRB.MINIMIZE) # 
                
                # constraints
                if t < T - 1:
                    m_backward[t][n][k].addConstr(theta_backward[t][n][k] >= theta_iniValue*T)
                    m_backward[t][n][k].addConstr(C_backward[t][n][k] >= theta_iniValue*T)
                    m_backward[t][n][k].addConstr(W1_backward[t][n][k] <= V)
                    m_backward[t][n][k].addConstr(W1_backward[t][n][k] + W2_backward[t][n][k] <= U)
                    m_backward[t][n][k].addConstr(C_backward[t][n][k] - vari_cost*q_backward[t][n][k] - W0_backward[t][n][k] + W1_backward[t][n][k] + W2_backward[t][n][k] + W3_backward[t][n][k] == overhead_cost[t])
                
                if t == 0:   
                    m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == ini_I + q_values[iter] - demand)
                else:
                    m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == I_forward_values[t-1][n] + q_forward_values[t-1][n] - demand)
                if t == 0:
                    m_backward[t][n][k].addConstr(C_backward[t][n][k] + price*B_backward[t][n][k] == ini_cash - overhead_cost[t] - vari_cost*q_values[iter]\
                                              -r1*W1_values[iter]-r2*W2_values[iter]-r3*W3_values[iter] + r0*W0_values[iter] + price*demand)
                else:
                    m_backward[t][n][k].addConstr(C_backward[t][n][k] + price*B_backward[t][n][k] == C_forward_values[t-1][n]- overhead_cost[t] - vari_cost*q_forward_values[t-1][n] -r1*W1_forward_values[t-1][n]\
                                              -r2*W2_forward_values[t-1][n] -r3*W3_forward_values[t-1][n]+ r0*W0_forward_values[t-1][n] + price*demand) 
      
                # optimize
                m_backward[t][n][k].optimize()                                   
                # m_backward[t][n][k].write('iter' + str(iter) + '_sub_' + str(t) + '^' + str(n) + '_' + str(k) +'-back.lp')
                # m_backward[t][n][k].write('iter' + str(iter) + '_sub_' + str(t) + '^' + str(n) + '_' + str(k) +'-back.sol')
                
                pi = m_backward[t][n][k].getAttr(GRB.Attr.Pi)
                rhs = m_backward[t][n][k].getAttr(GRB.Attr.RHS)
                
                num_con = len(pi)
                if t < T - 1:  
                    for kk in range(num_con - 2):
                        pi_rhs_value[t][n][k] += pi[kk]*rhs[kk]
                    pi_rhs_value[t][n][k] += -pi[-2] * demand - pi[-1]*overhead_cost[t] + pi[-1]*price*demand - price*demand + overhead_cost[t]
                    if t == 0:
                        pi_rhs_value[t][n][k] += pi[-1] * ini_cash
                        pi_rhs_value[t][n][k] += pi[-2] * ini_I
                else:
                    pi_rhs_value[t][n][k] = -pi[-2] * demand - pi[-1]*overhead_cost[t] + pi[-1]*price*demand - price*demand 
                pi_values[t][n][k] = [pi[-2], pi[-1]]
            
            arr = np.array(pi_values[t][n])    
            avg_pi = np.mean(arr, axis = 0)
            avg_pi_rhs = sum(pi_rhs_value[t][n]) / K
            
            slopes1[t][n].append(avg_pi[0])
            slopes2[t][n].append(avg_pi[1])
            intercept[t][n].append(avg_pi_rhs) 
            
                        
    iter += 1
    pass

end = time.process_time()
print('********************************************')
print('final expected total profits is %.2f', -z)
print('ordering Q in the first peiod is %.2f' % q_values[iter-1])
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)
