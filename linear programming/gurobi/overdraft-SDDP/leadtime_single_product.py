#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:18:39 2024

@author: zhenchen

@disp:  business overdraft for lead time in single product problem;

longer iterations seems more important than longer N;

    
    
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

theta_value = 0 

# cuts
slope1_stage = []
intercept1_stage = []
slopes1 = [[ [] for n in range(N)] for t in range(T) for i in range(iter_num)]
slopes2 = [[[0 for n in range(N)] for t in range(T)] for i in range(iter_num)]
slopes3 = [[[0 for n in range(N)] for t in range(T)] for i in range(iter_num)]
intercepts = [[ [] for n in range(N)] for t in range(T-1)]
q_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
qpre_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
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
        m.addConstr(theta >= slope1_stage[-1][0]*(ini_I) + slope1_stage[-1][1]*(ini_cash-vari_cost*q) + intercept1_stage[-1]) + slope1_stage[-1][2]*q
    m.update()
    m.optimize()
    
    q_values[iter][0] = [q.x for n in range(N)]     
    # m.write('iter' + str(iter+1) + '_main2.lp')    
    # m.write('iter' + str(iter+1) + '_main2.sol')
    
    W0_values[iter] = W0.x
    W1_values[iter] = W1.x
    W2_values[iter] = W2.x
    W3_values[iter] = W3.x
    theta_value = theta.x
    z = m.objVal    
    
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    q_pre_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'qpre_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    I_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    cash_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1)+ '^' + str(n+1)) for n in range(N)] for t in range(T)]
    W0_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W1_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W2_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W3_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W3_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    
    # B is the quantity of lost sale
    B_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
 
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
            
            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] + price*B_forward[t][n] == ini_cash - overhead_cost[t] - vari_cost*q_values[iter][t][n]\
                                          -r1*W1_values[iter] + r0*W0_values[iter]\
                                              -r2*W2_values[iter]-r3*W3_values[iter]+ price*demand)
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] + qpre_values[iter][t-1][n] - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] == cash_forward_values[t-1][n] - overhead_cost[t] \
                                                 - vari_cost*qpre_values[iter][t-1][n]\
                                                     -r1*W1_forward_values[t-1][n] + r0*W0_forward_values[t-1][n]\
                                                         -r2*W2_values[iter]-r3*W3_values[iter] + price*(demand - B_forward[t][n]))
            m_forward[t][n].addConstr(q_pre_forward[t][n] == q_values[iter][t][n])        
                       
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
            
            # put those cuts in the back
            if iter > 0 and t < T - 1:
                for i in range(iter):
                    for nn in range(1): # N
                        m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[i][t][nn]*(I_forward[t][n]+ q_pre_forward[t][n])\
                                                  + slopes3[i][t][nn]*q_forward[t][n]\
                                                  + slopes2[i][t][nn]*(cash_forward[t][n]- vari_cost*q_forward[t][n] - r3*W3_forward[t][n]-r2*W2_forward[t][n]\
                                             - r1*W1_forward[t][n]+r0*W0_forward[t][n])\
                                                + intercepts[t][nn][i])
            
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
                q_values[iter][t+1][n] = q_forward[t][n].x
                qpre_values[iter][t][n] = q_pre_forward[t][n].x
                theta_forward_values[t][n] = theta_forward[t][n].x
                W1_forward_values[t][n] = W1_forward[t][n].x
                W0_forward_values[t][n] = W0_forward[t][n].x
                W2_forward_values[t][n] = W2_forward[t][n].x
                W3_forward_values[t][n] = W3_forward[t][n].x
                
        # backward
        m_backward = [[[Model() for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
        q_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
        q_pre_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'qpre_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T)] 
        
        I_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T)]
        cash_backward = [[[m_backward[t][n][s].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T)]    
        W0_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
        W1_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
        W2_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
        W3_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W3_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
        
        # B is the quantity of lost sale
        B_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
        theta_backward = [[[m_backward[t][n][s].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for s in range(sample_nums[t])] for n in range(N)] for t in range(T - 1)]
        
        intercept_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
        slope1_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
        slope2_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
        
        for t in range(T-1, -1, -1):    
            for n in range(N):      
                S = len(sample_detail[t])
                for s in range(S):
                    demand = sample_detail[t][s]
                    
                    if t == 0:   
                        m_backward[t][n][s].addConstr(I_backward[t][n][s] - B_backward[t][n][s] == ini_I - demand)
                        m_backward[t][n][s].addConstr(cash_backward[t][n][s] == ini_cash - overhead_cost[t] - vari_cost*q_values[iter][t][n]\
                                                      -r3*W3_values[iter] - r2*W2_values[iter] - r1*W1_values[iter] + r0*W0_values[iter] + price*(demand - B_backward[t][n][s]))
                    else:
                        m_backward[t][n][s].addConstr(I_backward[t][n][s] - B_backward[t][n][s] == I_forward_values[t-1][n] + qpre_values[iter][t-1][n] - demand)
                        m_backward[t][n][s].addConstr(cash_backward[t][n][s] == cash_forward_values[t-1][n]- overhead_cost[t]\
                                                      - vari_cost*qpre_values[iter][t-1][n]\
                                                      - r3*W3_forward_values[t-1][n]- r2*W2_forward_values[t-1][n]- r1*W1_forward_values[t-1][n]\
                                                      + r0*W0_forward_values[t-1][n] + price*(demand - B_backward[t][n][s]))
                    
                    m_backward[t][n][s].addConstr(q_pre_backward[t][n] == q_values[iter][t][n])        
                    
                    if t == T - 1:                   
                        m_backward[t][n][s].setObjective(-price*(demand - B_backward[t][n][s]) - unit_salvage*I_backward[t][n][s], GRB.MINIMIZE)
                    else:
                        m_backward[t][n][s].setObjective(overhead_cost[t] + vari_cost*q_backward[t][n][s] - price*(demand - B_backward[t][n][s])\
                                                         + r3*W3_backward[t][n][s] + r2*W2_backward[t][n][s]
                                                         + r1*W1_backward[t][n][s] - r0*W0_backward[t][n][s] + theta_backward[t][n][s], GRB.MINIMIZE)  
                        m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= theta_iniValue*(T-1-t))
                        m_backward[t][n][s].addConstr(W1_backward[t][n][s] <= V)
                        m_backward[t][n][s].addConstr(W1_backward[t][n][s] + W2_backward[t][n][s] <= U)
                        m_backward[t][n][s].addConstr(cash_backward[t][n][s] - vari_cost*q_backward[t][n][s] - W0_backward[t][n][s]\
                                                      + W1_backward[t][n][s] + W2_backward[t][n][s] + W3_backward[t][n][s]== overhead_cost[t])
                                
                    # put those cuts in the back
                    if iter > 0 and t < T - 1:
                        for i in range(iter):
                            for nn in range(1): # N
                                m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= slopes1[i][t][nn]*q_forward[t][n]+\
                                                              slopes2[t][nn][i][-2]*(I_backward[t][n][s]+ q_backward[t][n][s]) + slopes[t][nn][i][-1]*(cash_backward[t][n][s]- vari_cost*q_backward[t][n][s]\
                                                                                       -r3*W3_backward[t][n][s]-r2*W2_backward[t][n][s]\
                                                                                          - r1*W1_backward[t][n][s] + r0*W0_backward[t][n][s]) + intercepts[t][nn][i])
                                   
                    # optimize
                    m_backward[t][n][s].optimize()
                    pi = m_backward[t][n][s].getAttr(GRB.Attr.Pi)
                    # m_backward[t][n][s].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '-' + str(k+1) +'back.lp')
                    
                    rhs = m_backward[t][n][s].getAttr(GRB.Attr.RHS)
                    num_con = len(pi)
                    
                    
                    if t < T - 1:
                        intercept_values[t][n][s] += -pi[0]*demand + pi[1]*price*demand - pi[1]*overhead_cost[t] - price*demand + overhead_cost[t+1] # put here is better because of demand
                    else:
                        intercept_values[t][n][s] += -pi[0]*demand + pi[1]*price*demand - pi[1]*overhead_cost[t] - price*demand # put here is better because of demand
                    for sk in range(2, num_con):
                        intercept_values[t][n][s] += pi[kk]*rhs[kk]
                    
                    slope1_values[t][n][s] = pi[0]                                                         
                    slope2_values[t][n][s] = pi[1]
                    slope3_values[t][n][s] = pi[2]
                
                avg_intercept = sum(intercept_values[t][n]) / S
                avg_slope1 = sum(slope1_values[t][n]) / S
                avg_slope2 = sum(slope2_values[t][n]) / S
                avg_slope3 = sum(slope3_values[t][n]) / S
                if t == 0:
                    slope1_stage[iter] = avg_slope1
                    intercept1_stage[iter] = avg_intercept
                else:
                    slopes1[iter][t-1][n] = avg_slope1 
                    slopes2[iter][t-1][n] = avg_slope2               
                    intercepts[iter][t-1][n] = avg_intercept  

    iter += 1

end = time.process_time()
print('********************************************')
print('final expected total costs is %.2f' % z)
print('ordering Q in the first peiod is %.2f' % q_values[iter-1][0][0])
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)        
               
    
    
    

