#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:52:47 2023

@author: zhenchen

@disp:  
    
    constraints of benders cuts are in the back;    
    longer iterations seems more important than longer N;
    
    lower bound for the variables must in the constraints, not in the defintion of the variables;
    if rhs in one constraint has the previous stage decision variable, then the rhs_value must be
    computed separately;
    
    larger T results in larger gaps, more samples/iterations results in smaller gaps.
    more decision variables require more samples.
    
    ini_I = 0
    ini_cash = 0
    vari_cost = 1
    price = 10
    unit_back_cost = 0
    unit_hold_cost = 0
    unit_salvage = 0.5
    mean_demands = [10, 15, 10]
    T = len(mean_demands)
    sample_nums = [10 for t in range(T)]
    overhead_cost = [50 for t in range(T)]

    r0 = 0.01
    r1 = 0
    r2 = 0.1
    r3 = 1 # penalty interest rate for overdraft exceeding the limit
    V = 20 # free-interest limit
    U = 100 # overdraft limit
    
    3 periods with 60 samples and 15 iterations, gap very close to optimal, takes over 400s;
    4 periods with 50 samples and 13 iterations, gap very close to optimal, takes time about 990s;
    
    for parameters:
        mean_demands = [20, 35, 20]
        V = 200 # free-interest limit
        U = 1000 # overdraft limit
        iter_num = 15
        N = 20 # sampled number of scenarios for forward computing
        
        SDP result 371, Q1=39
        SDDP result 370.09(N=1), Q1=27, cpu time 73s;
        SDDP result 384.82, Q1=33, cpu time 121s;
        SDDP result 374.82, Q1=26, cpu time 181s;
        overhead_cost = [100 for t in range(T)];
"""

from gurobipy import *
import itertools
import random
import time
import numpy as np

import sys 
sys.path.append("..") 
from tree import generate_sample, get_tree_strcture, generate_scenario_samples



start = time.process_time()
ini_I = 0
ini_cash = 0
vari_cost = 1
price = 10
unit_back_cost = 0
unit_hold_cost = 0
unit_salvage = 0.5
mean_demands = [20, 35, 20]
T = len(mean_demands)
sample_nums = [10 for t in range(T)]
overhead_cost = [100 for t in range(T)]

r0 = 0.01
r1 = 0
r2 = 0.1
r3 = 1 # penalty interest rate for overdraft exceeding the limit
V = 200 # free-interest limit
U = 1000 # overdraft limit

trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = 1
for i in sample_nums:
    scenario_numTotal *= i

# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    sample_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
# sample_detail = [[5, 15], [5, 15], [15, 5], [15, 15]]
scenarios_full = list(itertools.product(*sample_detail)) 

iter = 0
iter_num = 15
N = 20 # sampled number of scenarios for forward computing

theta_iniValue = -2000 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
W0 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^0')
W1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^1')
W2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^2')
W3 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^3')
theta = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_2')
m.setObjective(overhead_cost[0] + vari_cost*q + r3*W3 + r2*W2 + r1*W1 - r0*W0 + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))
m.addConstr(W1 <= V)
m.addConstr(W1 + W2 <= U)
m.addConstr(-vari_cost*q - W0 + W1 + W2 + W3 == overhead_cost[0] - ini_cash)
# m.addConstr(vari_cost * q <= ini_cash)

theta_value = 0 

# cuts
slope1_stage = []
intercept1_stage = []
slopes = [[ [] for n in range(N)] for t in range(T-1)]
intercepts = [[ [] for n in range(N)] for t in range(T-1)]
q_values = [0 for iter in range(iter_num)]
W0_values = [0 for iter in range(iter_num)]
W1_values = [0 for iter in range(iter_num)]
W2_values = [0 for iter in range(iter_num)]
W3_values = [0 for iter in range(iter_num)]

start = time.process_time()
while iter < iter_num:  
    
    # sample a numer of scenarios from the full scenario tree
    # random.seed(10000)
    sample_scenarios = generate_scenario_samples(N, trunQuantile, mean_demands)
    # sample_scenarios = [[5, 5], [5, 15], [15, 5], [15, 15]]
    sample_scenarios.sort() # sort to make same numbers together
    
    # forward
    if iter > 0:
        m.addConstr(theta >= slope1_stage[-1][-2]*(ini_I+q) + slope1_stage[-1][-1]*(ini_cash-vari_cost*q) + intercept1_stage[-1])
    m.update()
    m.optimize()
      
    q_values[iter] = q.x    
    if iter < 4:
        m.write('iter' + str(iter+1) + '_main1.lp')    
        pass
    
    W0_values[iter] = W0.x
    W1_values[iter] = W1.x
    W2_values[iter] = W2.x
    W3_values[iter] = W3.x
    theta_value = theta.x
    z = m.objVal    
    
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    I_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    cash_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1)+ '^' + str(n+1)) for n in range(N)] for t in range(T)]
    W0_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W1_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W2_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W3_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W3_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    
    # B is the quantity of lost sale
    B_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]

    q_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    theta_forward_values = [[0 for n in range(N)] for t in range(T)]
    W0_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    W1_forward_values = [[0 for n in range(N)] for t in range(T-1)]
    W2_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    W3_forward_values = [[0 for n in range(N)] for t in range(T-1)]
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]
                                  
            if t == T - 1:                   
                m_forward[t][n].setObjective(-price*(demand - B_forward[t][n]) - unit_salvage*I_forward[t][n], GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(overhead_cost[t] + vari_cost*q_forward[t][n] - price*(demand - B_forward[t][n])\
                                             + r3*W3_forward[t][n] + r2*W2_forward[t][n]\
                                             + r1*W1_forward[t][n] - r0*W0_forward[t][n] + theta_forward[t][n], GRB.MINIMIZE)  
                m_forward[t][n].addConstr(theta_forward[t][n] >= theta_iniValue*(T-1-t))
                m_forward[t][n].addConstr(cash_forward[t][n] - vari_cost*q_forward[t][n] - W0_forward[t][n]\
                                          + W1_forward[t][n] + W2_forward[t][n] + W3_forward[t][n]== overhead_cost[t])
                m_forward[t][n].addConstr(W1_forward[t][n] <= V)
                m_forward[t][n].addConstr(W1_forward[t][n] + W2_forward[t][n] <= U)
                
                # m_forward[t][n].addConstr(vari_cost * q_forward[t][n] <= cash_forward[t][n])
            # m_forward[t][n].addConstr(B_forward[t][n] <= demand) # not necessary

            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I + q_values[iter] - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] + price*B_forward[t][n] == ini_cash - overhead_cost[t] - vari_cost*q_values[iter]\
                                          -r1*W1_values[iter] + r0*W0_values[iter]\
                                              -r2*W2_values[iter]-r3*W3_values[iter]+ price*demand)
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] + q_forward_values[t-1][n] - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] == cash_forward_values[t-1][n] - overhead_cost[t] \
                                                 - vari_cost*q_forward_values[t-1][n]\
                                                     -r1*W1_forward_values[t-1][n] + r0*W0_forward_values[t-1][n]\
                                                         -r2*W2_forward_values[t-1][n]-r3*W3_forward_values[t-1][n] + price*(demand - B_forward[t][n]))
            # put those cuts in the back, it does not matter whether in back or front
            if iter > 0 and t < T - 1:
                for i in range(iter):
                    for nn in range(N): # N
                        m_forward[t][n].addConstr(theta_forward[t][n] >= slopes[t][nn][i][-2]*(I_forward[t][n]+ q_forward[t][n]) + slopes[t][nn][i][-1]*(cash_forward[t][n]- vari_cost*q_forward[t][n]\
                                                                            -r3*W3_forward[t][n]-r2*W2_forward[t][n]\
                                                                                - r1*W1_forward[t][n]+r0*W0_forward[t][n]) + intercepts[t][nn][i])
               
            # optimize
            m_forward[t][n].optimize()
            try:
                I_forward_values[t][n] = I_forward[t][n].x 
            except:
                m_forward[t][n].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '.lp') 
                pass
            B_forward_values[t][n] = B_forward[t][n].x  
            cash_forward_values[t][n] = cash_forward[t][n].x 
            if t < T - 1:
                q_forward_values[t][n] = q_forward[t][n].x
                theta_forward_values[t][n] = theta_forward[t][n].x
                W1_forward_values[t][n] = W1_forward[t][n].x
                W0_forward_values[t][n] = W0_forward[t][n].x
                W2_forward_values[t][n] = W2_forward[t][n].x
                W3_forward_values[t][n] = W3_forward[t][n].x
    
    # backward
    m_backward = [[[Model() for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    q_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    I_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T)]
    cash_backward = [[[m_backward[t][n][k].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T)]    
    W0_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    W1_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    W2_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    W3_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'W3_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    
    # B is the quantity of lost sale
    B_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward = [[[m_backward[t][n][k].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T - 1)]

    q_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    I_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    cash_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    B_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    pi_values = [[[[]  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    pi_rhs_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)] 

    for t in range(T-1, -1, -1):    
        for n in range(N):      
            K = len(sample_detail[t])
            for k in range(K):
                demand = sample_detail[t][k]
                # put those cuts in the front, may be put front is better for generating cuts
                if iter > 0 and t < T - 1:
                    for i in range(iter):
                        for nn in range(N): # N
                            m_backward[t][n][k].addConstr(theta_backward[t][n][k] >= slopes[t][nn][i][-2]*(I_backward[t][n][k]+ q_backward[t][n][k]) + slopes[t][nn][i][-1]*(cash_backward[t][n][k]- vari_cost*q_backward[t][n][k]\
                                                                                   -r3*W3_backward[t][n][k]-r2*W2_backward[t][n][k]\
                                                                                      - r1*W1_backward[t][n][k] + r0*W0_backward[t][n][k]) + intercepts[t][nn][i])
               
                
                if t == T - 1:                   
                    m_backward[t][n][k].setObjective(-price*(demand - B_backward[t][n][k]) - unit_salvage*I_backward[t][n][k], GRB.MINIMIZE)
                else:
                    m_backward[t][n][k].setObjective(overhead_cost[t] + vari_cost*q_backward[t][n][k] - price*(demand - B_backward[t][n][k])\
                                                     + r3*W3_backward[t][n][k] + r2*W2_backward[t][n][k]
                                                     + r1*W1_backward[t][n][k] - r0*W0_backward[t][n][k] + theta_backward[t][n][k], GRB.MINIMIZE)  
                    m_backward[t][n][k].addConstr(theta_backward[t][n][k] >= theta_iniValue*(T-1-t))
                    m_backward[t][n][k].addConstr(W1_backward[t][n][k] <= V)
                    m_backward[t][n][k].addConstr(W1_backward[t][n][k] + W2_backward[t][n][k] <= U)
                    m_backward[t][n][k].addConstr(cash_backward[t][n][k] - vari_cost*q_backward[t][n][k] - W0_backward[t][n][k]\
                                                  + W1_backward[t][n][k] + W2_backward[t][n][k] + W3_backward[t][n][k]== overhead_cost[t])
                
                    # m_backward[t][n][k].addConstr(vari_cost * q_backward[t][n][k] <= cash_backward[t][n][k])
                # m_backward[t][n][k].addConstr(B_backward[t][n][k] <= demand)

                if t == 0:   
                    m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == ini_I + q_values[iter] - demand)
                    m_backward[t][n][k].addConstr(cash_backward[t][n][k] == ini_cash - overhead_cost[t] - vari_cost*q_values[iter]\
                                                  -r3*W3_values[iter] - r2*W2_values[iter] - r1*W1_values[iter] + r0*W0_values[iter] + price*(demand - B_backward[t][n][k]))
                else:
                    m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == I_forward_values[t-1][n] + q_forward_values[t-1][n] - demand)
                    m_backward[t][n][k].addConstr(cash_backward[t][n][k] == cash_forward_values[t-1][n]- overhead_cost[t]\
                                                  - vari_cost*q_forward_values[t-1][n]\
                                                  - r3*W3_forward_values[t-1][n]- r2*W2_forward_values[t-1][n]- r1*W1_forward_values[t-1][n]\
                                                  + r0*W0_forward_values[t-1][n] + price*(demand - B_backward[t][n][k]))
                
                # optimize
                m_backward[t][n][k].optimize()                
                # if t == 0 and n == 0 and iter > 0:
                #     m_backward[t][n][k].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '-' + str(k+1) +'back.lp')
                try:
                    pi = m_backward[t][n][k].getAttr(GRB.Attr.Pi)
                except Exception:
                    m_backward[t][n][k].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '-' + str(k+1) +'back.lp')
                    pass
                
                rhs = m_backward[t][n][k].getAttr(GRB.Attr.RHS)
                num_con = len(pi)
                for kk in range(num_con-2):
                    pi_rhs_values[t][n][k] += pi[kk]*rhs[kk]
                # demand should put here, can not put in the above rhs, 
                # rhs may be wrong because it have previous stage decision variable
                if t < T - 1:
                    pi_rhs_values[t][n][k] += -pi[-2]*demand + pi[-1]*price*demand - pi[-1]*overhead_cost[t] - price*demand + overhead_cost[t+1] # put here is better because of demand
                else:
                    pi_rhs_values[t][n][k] += -pi[-2]*demand + pi[-1]*price*demand - pi[-1]*overhead_cost[t] - price*demand # put here is better because of demand
                pi_values[t][n][k].append(pi)
             
            avg_pi = sum(np.array(pi_values[t][n])) / K
            avg_pi_rhs = sum(pi_rhs_values[t][n]) / K
                
            # recording cuts
            if t == 0 and n == 0:
                slope1_stage.append(avg_pi[0])
                intercept1_stage.append(avg_pi_rhs)
                if iter == 1:
                    pass
            elif t > 0:
                slopes[t-1][n].append(avg_pi[0])
                intercepts[t-1][n].append(avg_pi_rhs)   
            print()

    iter += 1

end = time.process_time()
print('********************************************')
final_cash = -z
print('final expected cash increment is %.2f' % final_cash)
print('ordering Q in the first peiod is %.2f' % q.x)
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)


