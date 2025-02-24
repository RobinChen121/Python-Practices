#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:52:47 2023

@author: zhenchen

@disp:  
    
    lower bound for the variables must in the constraints, not in the defintion of the variables;
    if rhs in one constraint has the previous stage decision variable, then the rhs_value must be
    computed separately;
    
    larger T results in larger gaps, more samples/iterations results in smaller gaps.
    
ini_I = 0
ini_cash = 0
vari_cost = 1
price = 10
unit_back_cost = 0
unit_hold_cost = 0
mean_demands = [10, 10, 10]
T = len(mean_demands)
sample_nums = [10 for t in range(T)]
iter_num = 11
N = 10 # sampled number of scenarios for forward computing

SDP results is 274, Q = 24;
SDDP results is 268.7, Q = 13.8;

SDDP with abridge:
for abridge, B=2, F=5 (take average for the 5 out of the 10 solutions of forward computing), so compute 10 sub problems in forward;
final expected cash increment is 265.70
ordering Q in the first peiod is 30.00
cpu time is 6.434 s;
final expected cash increment is 261.40
ordering Q in the first peiod is 15.00
cpu time is 7.448 s

"""

from gurobipy import *
import itertools
import random
import time
import numpy as np

import sys
sys.path.append("..") 
from tree import *



start = time.process_time()
ini_I = 0
ini_cash = 0
vari_cost = 1
price = 10
unit_back_cost = 0
unit_hold_cost = 0
mean_demands = [10, 10, 10]
T = len(mean_demands)
sample_nums = [10 for t in range(T)] # samples in one stage
iter_num = 15
N = 10 # sampled number of scenarios for forward computing

trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = 1
for i in sample_nums:
    scenario_numTotal *= i

# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    sample_detail[t] = generate_samples(sample_nums[t], trunQuantile, mean_demands[t])
# sample_detail = [[5, 15], [5, 15], [15, 5], [15, 15]]
scenarios_full = list(itertools.product(*sample_detail)) 


iter = 0



theta_iniValue = -500 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_2')
m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))
# m.addConstr(vari_cost * q <= ini_cash)

theta_value = 0 

# cuts
slope1_stage = []
intercept1_stage = []
slopes = [[ [] for n in range(N)] for t in range(T-1)]
intercepts = [[ [] for n in range(N)] for t in range(T-1)]
q_values = [0 for iter in range(iter_num)]
q_sub_values = [[[0 for n in range(N)] for t in range(T-1)] for iter in range(iter_num)]

B = 10 # number of sampled nodes in forwad subproblems for abridge

time_start = time.process_time()
while iter < iter_num:  
    z_values = [[0 for t in range(T)] for n in range(N)]
    
    # sample a numer of scenarios from the full scenario tree
    # random.seed(10000)
    sample_scenarios = generate_scenarios(N, trunQuantile, mean_demands)
    # sample_scenarios = [[5, 5], [5, 15], [15, 5], [15, 15]]
    # sample_scenarios = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [15,5,5], [15,15,5], [15,5, 15], [5,15,15],[15,15,15]]
    sample_scenarios.sort() # sort to make same numbers together
    
    # forward
    if iter > 0:
        m.addConstr(theta >= slope1_stage[-1][-2]*(ini_I+q) + slope1_stage[-1][-1]*(ini_cash-vari_cost*q) + intercept1_stage[-1])
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    # if iter == 2:
    #     m.write('iter' + str(iter+1) + '_main2.lp') 
    #     m.write('iter' + str(iter+1) + '_main2.sol')        
    #     pass

    
    q_values[iter] = q.x
    theta_value = theta.x
    z = m.objVal
    
    
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    I_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    cash_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1)+ '^' + str(n+1)) for n in range(N)] for t in range(T)]
    
    # B is the quantity of lost sale
    B_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]

    q_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    
    
    q_forward_values_bridge = [[0 for n in range(B)] for t in range(T-1)] 
    I_forward_values_bridge = [[0 for n in range(B)] for t in range(T)]
    B_forward_values_bridge = [[0 for n in range(B)] for t in range(T)]
    cash_forward_values_bridge = [[0 for n in range(B)] for t in range(T)]
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]
            bn = int(n /(N/B))
            
            # put those cuts in the front
            if iter > 0 and t < T - 1:
                for i in range(iter):
                    for nn in range(B): # N
                        m_forward[t][n].addConstr(theta_forward[t][n] >= slopes[t][nn][i][-2]*(I_forward[t][n]+ q_forward[t][n]) + slopes[t][nn][i][-1]*(cash_forward[t][n]- vari_cost*q_forward[t][n]) + intercepts[t][nn][i])
           
            
            if t == T - 1:                   
                m_forward[t][n].setObjective(-price*(demand - B_forward[t][n]), GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(vari_cost*q_forward[t][n] - price*(demand - B_forward[t][n]) + theta_forward[t][n], GRB.MINIMIZE)  
                m_forward[t][n].addConstr(theta_forward[t][n] >= theta_iniValue*(T-1-t))
                # m_forward[t][n].addConstr(vari_cost * q_forward[t][n] <= cash_forward[t][n])
            # m_forward[t][n].addConstr(B_forward[t][n] <= demand) # not necessary
            # if t == 0:
            #     m_forward[t][n].addConstr(B_forward[t][n] >= demand - ini_I - q_values[iter])
            # else:
            #     m_forward[t][n].addConstr(B_forward[t][n] >= demand - I_forward_values[t-1][n] - q_forward_values[t-1][n])

            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I + q_values[iter] - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] == ini_cash - vari_cost*q_values[iter] + price*(demand - B_forward[t][n]))
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values_bridge[t-1][bn] + q_forward_values_bridge[t-1][bn] - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] == cash_forward_values_bridge[t-1][bn]- vari_cost*q_forward_values_bridge[t-1][bn] + price*(demand - B_forward[t][n]))

                
            # optimize
            m_forward[t][n].Params.LogToConsole = 0
            m_forward[t][n].optimize()
            I_forward_values[t][n] = I_forward[t][n].x 
            
            if t < T - 1:
                z_values[n][t] = -m_forward[t][n].objVal + theta_forward[t][n].x
            else:
                z_values[n][t] = -m_forward[t][n].objVal

            B_forward_values[t][n] = B_forward[t][n].x  
            cash_forward_values[t][n] = cash_forward[t][n].x 
            if t < T - 1:
                q_forward_values[t][n] = q_forward[t][n].x
        for bb in range(B):
            start = int(bb*N/B)
            end = int((bb+1)*N/B)
            I_forward_values_bridge[t][bb] = np.mean(I_forward_values[t][start : end])
            B_forward_values_bridge[t][bb] = np.mean(B_forward_values[t][start : end])
            cash_forward_values_bridge[t][bb] = np.mean(cash_forward_values[t][start : end])
            if t < T - 1:
                q_forward_values_bridge[t][bb] = np.mean(q_forward_values[t][start : end])
    
    # backward
    m_backward = [[[Model() for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    q_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    I_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T)]
    cash_backward = [[[m_backward[t][n][k].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T)]    
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
        for bn in range(B):      
            K = len(sample_detail[t])
            for k in range(K):
                demand = sample_detail[t][k]
                # put those cuts in the front
                if iter > 0 and t < T - 1:
                    for i in range(iter):
                        for nn in range(B): # N
                            m_backward[t][bn][k].addConstr(theta_backward[t][bn][k] >= slopes[t][nn][i][-2]*(I_backward[t][bn][k]+ q_backward[t][bn][k]) + slopes[t][nn][i][-1]*(cash_backward[t][bn][k]- vari_cost*q_backward[t][bn][k]) + intercepts[t][nn][i])
               
                
                if t == T - 1:                   
                    m_backward[t][bn][k].setObjective(-price*(demand - B_backward[t][bn][k]), GRB.MINIMIZE)
                else:
                    m_backward[t][bn][k].setObjective(vari_cost*q_backward[t][bn][k] - price*(demand - B_backward[t][bn][k]) + theta_backward[t][bn][k], GRB.MINIMIZE)  
                    m_backward[t][bn][k].addConstr(theta_backward[t][bn][k] >= theta_iniValue*(T-1-t))
                    # m_backward[t][bn][k].addConstr(vari_cost * q_backward[t][bn][k] <= cash_backward[t][bn][k])
                # m_backward[t][bn][k].addConstr(B_backward[t][bn][k] <= demand)
                # if t == 0:
                #     m_backward[t][bn][k].addConstr(B_backward[t][bn][k] >= demand - ini_I - q_values[iter])
                # else:
                #     m_backward[t][bn][k].addConstr(B_forward[t][bn][k] >= demand - I_forward_values[t-1][bn] - q_forward_values[t-1][bn])
                    
                if t == 0:   
                    m_backward[t][bn][k].addConstr(I_backward[t][bn][k] - B_backward[t][bn][k] == ini_I + q_values[iter] - demand)
                    m_backward[t][bn][k].addConstr(cash_backward[t][bn][k] == ini_cash - vari_cost*q_values[iter] + price*(demand - B_backward[t][bn][k]))
                else:
                    m_backward[t][bn][k].addConstr(I_backward[t][bn][k] - B_backward[t][bn][k] == I_forward_values_bridge[t-1][bn] + q_forward_values_bridge[t-1][bn] - demand)
                    m_backward[t][bn][k].addConstr(cash_backward[t][bn][k] == cash_forward_values_bridge[t-1][bn]- vari_cost*q_forward_values_bridge[t-1][bn] + price*(demand - B_backward[t][bn][k]))
                
                # optimize
                m_backward[t][bn][k].Params.LogToConsole = 0
                m_backward[t][bn][k].optimize()                
                pi = m_backward[t][bn][k].getAttr(GRB.Attr.Pi)
                # if t == 0 and bn == 0 and iter == 3:
                #     m_backward[t][bn][k].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(bn+1) + '-' + str(k+1) +'back2.lp')
                #     m_backward[t][bn][k].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(bn+1) + '-' + str(k+1) +'back2.lol')               
                #     pass

                
                
                rhs = m_backward[t][bn][k].getAttr(GRB.Attr.RHS)
                num_con = len(pi)
                for kk in range(num_con-2): # range(num_con-2):
                    pi_rhs_values[t][bn][k] += pi[kk]*rhs[kk]
                # demand should put here, can not put in the above rhs, 
                # rhs may be wrong because it have previous stage decision variable
                pi_rhs_values[t][bn][k] += -pi[-2] * demand + pi[-1] * price * demand - price*demand # put here is better because of demand
                # pi_rhs_values[t][bn][k] += pi[-3] * demand
                pi_values[t][bn][k].append(pi)
            
   
            avg_pi = sum(np.array(pi_values[t][bn])) / K
            avg_pi_rhs = sum(pi_rhs_values[t][bn]) / K
            if iter > 0 and t == 0 and n == 0:
                pass
                
            # recording cuts
            if t == 0 and bn == 0:
                slope1_stage.append(avg_pi[0])
                intercept1_stage.append(avg_pi_rhs)
                if iter == 1:
                    pass
            elif t > 0:
                slopes[t-1][bn].append(avg_pi[0])
                intercepts[t-1][bn].append(avg_pi_rhs)   
    
    # z_lb, z_ub = compute_ub(z_values)
    # if -z <= z_ub and -z >= z_lb:
    #     print('********************************************')
    #     print('iteration ends in iter + 1 = %d' % iter)
    #     break
    iter += 1

time_end = time.process_time()
print('********************************************')
final_cash = -z
print('final expected cash increment is %.2f' % final_cash)
print('ordering Q in the first peiod is %.2f' % q.x)
cpu_time = time_end - time_start
print('cpu time is %.3f s' % cpu_time)


