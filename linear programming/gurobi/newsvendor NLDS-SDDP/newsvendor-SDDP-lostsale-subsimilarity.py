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
SDDP results is 268.7, Q = 13.8; cpu time 62s (windows desktop);
**************************************

leverage the subproblem similarity to speed up the computation:
    

"""

from gurobipy import *
import itertools
import random
import time
import numpy as np

sys.path.append("..") 
from tree import generate_sample, generate_scenario_samples, compute_ub



start = time.process_time()
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

start = time.process_time()
while iter < iter_num:  
    z_values = [[0 for t in range(T)] for n in range(N)]
    
    # sample a numer of scenarios from the full scenario tree
    # random.seed(10000)
    sample_scenarios = generate_scenario_samples(N, trunQuantile, mean_demands)
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
    #     m.write('iter' + str(iter+1) + '_main.lp') 
    #     m.write('iter' + str(iter+1) + '_main.sol')        
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
    theta_forward_values = [[0 for n in range(N)] for t in range(T)]
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]
            
            positive_computed_before = False
            negative_computed_before = False
            positive_store_values = []
            negative_store_values = []
           
            if positive_computed_before == False and negative_computed_before == False: 
                # put those cuts in the front
                if iter > 0 and t < T - 1:
                    for i in range(iter):
                        for nn in range(N): # N
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
                    if ini_I + q_values[iter] - demand > 0:
                        positive_computed_before = True
                    else:
                        negative_computed_before = True
                else:
                    m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] + q_forward_values[t-1][n] - demand)
                    m_forward[t][n].addConstr(cash_forward[t][n] == cash_forward_values[t-1][n]- vari_cost*q_forward_values[t-1][n] + price*(demand - B_forward[t][n]))
                    if I_forward_values[t-1][n] + q_forward_values[t-1][n] - demand > 0:
                        positive_computed_before = True
                    else:
                        negative_computed_before = True
                    
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
                    theta_forward_values[t][n] = theta_forward[t][n].x
                if positive_computed_before == True:               
                    if t < T - 1:
                        positive_store_values = [I_forward_values[t][n], z_values[n][t], B_forward_values[t][n], cash_forward_values[t][n], q_forward_values[t][n], theta_forward_values[t][n]]
                    else:
                        positive_store_values = [I_forward_values[t][n], z_values[n][t], B_forward_values[t][n], cash_forward_values[t][n]]
        
                elif negative_computed_before == True:                
                    if t < T - 1:
                        negative_store_values = [I_forward_values[t][n], z_values[n][t], B_forward_values[t][n], cash_forward_values[t][n], q_forward_values[t][n], theta_forward_values[t][n]]
                    else:
                        negative_store_values = [I_forward_values[t][n], z_values[n][t], B_forward_values[t][n], cash_forward_values[t][n]]
                
            elif positive_computed_before == True:
                I_forward_values[t][n], z_values[n][t], B_forward_values[t][n], cash_forward_values[t][n], q_forward_values[t][n], theta_forward_values[t][n] = positive_store_values
            else:
                I_forward_values[t][n], z_values[n][t], B_forward_values[t][n], cash_forward_values[t][n], q_forward_values[t][n], theta_forward_values[t][n] = negative_store_values
    
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
        for n in range(N):      
            K = len(sample_detail[t])
            
            positive_computed_before = False
            negative_computed_before = False
            positive_store_values = [None for i in range(2)]
            negative_store_values = [None for i in range(2)]
           
            for k in range(K):
                demand = sample_detail[t][k]
                
                if positive_computed_before == False and negative_computed_before == False:                
                    # put those cuts in the front
                    if iter > 0 and t < T - 1:
                        for i in range(iter):
                            for nn in range(N): # N
                                m_backward[t][n][k].addConstr(theta_backward[t][n][k] >= slopes[t][nn][i][-2]*(I_backward[t][n][k]+ q_backward[t][n][k]) + slopes[t][nn][i][-1]*(cash_backward[t][n][k]- vari_cost*q_backward[t][n][k]) + intercepts[t][nn][i])
                   
                    
                    if t == T - 1:                   
                        m_backward[t][n][k].setObjective(-price*(demand - B_backward[t][n][k]), GRB.MINIMIZE)
                    else:
                        m_backward[t][n][k].setObjective(vari_cost*q_backward[t][n][k] - price*(demand - B_backward[t][n][k]) + theta_backward[t][n][k], GRB.MINIMIZE)  
                        m_backward[t][n][k].addConstr(theta_backward[t][n][k] >= theta_iniValue*(T-1-t))
                    
                    # m_backward[t][n][k].addConstr(vari_cost * q_backward[t][n][k] <= cash_backward[t][n][k])
                    # m_backward[t][n][k].addConstr(B_backward[t][n][k] <= demand)
                    # if t == 0:
                    #     m_backward[t][n][k].addConstr(B_backward[t][n][k] >= demand - ini_I - q_values[iter])
                    # else:
                    #     m_backward[t][n][k].addConstr(B_forward[t][n][k] >= demand - I_forward_values[t-1][n] - q_forward_values[t-1][n])
                        
                    if t == 0:   
                        m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == ini_I + q_values[iter] - demand)
                        m_backward[t][n][k].addConstr(cash_backward[t][n][k] == ini_cash - vari_cost*q_values[iter] + price*(demand - B_backward[t][n][k]))
                        if ini_I + q_values[iter] - demand > 0:
                            positive_computed_before = True
                        else:
                            negative_computed_before = True
                    else:
                        m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == I_forward_values[t-1][n] + q_forward_values[t-1][n] - demand)
                        m_backward[t][n][k].addConstr(cash_backward[t][n][k] == cash_forward_values[t-1][n]- vari_cost*q_forward_values[t-1][n] + price*(demand - B_backward[t][n][k]))
                        if I_forward_values[t-1][n] + q_forward_values[t-1][n] - demand > 0:
                            positive_computed_before = True
                        else:
                            negative_computed_before = True
                    
                    # optimize
                    m_backward[t][n][k].Params.LogToConsole = 0
                    m_backward[t][n][k].optimize()                
                    pi = m_backward[t][n][k].getAttr(GRB.Attr.Pi)
                    # if t == 0 and n == 0 and iter == 1:
                    #     m_backward[t][n][k].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '-' + str(k+1) +'back.lp')
                    #     pass
                    
                    
                    rhs = m_backward[t][n][k].getAttr(GRB.Attr.RHS)
                    pi_values[t][n][k].append(pi)
                    if positive_computed_before == True:
                        positive_store_values = [pi, rhs]
                    elif negative_computed_before == True:
                        negative_store_values = [pi, rhs]
                elif positive_computed_before == True:
                    pi_values[t][n][k].append(positive_store_values[0])
                    pi = positive_store_values[0]
                    rhs = positive_store_values[1]                   
                else:
                    pi_values[t][n][k].append(negative_store_values[0])
                    pi = negative_store_values[0]
                    rhs = negative_store_values[1]
                # not necessary
                # rhs[-2] = ini_I + q_values[iter] - demand if t == 0 else I_forward_values[t-1][n] + q_forward_values[t-1][n] - demand
                # rhs[-1] = ini_cash - vari_cost*q_values[iter] + price*demand if t == 0 else cash_forward_values[t-1][n]- vari_cost*q_forward_values[t-1][n] + price*demand
               
                num_con = len(pi)
                for kk in range(num_con-2): # range(num_con-2):
                    pi_rhs_values[t][n][k] += pi[kk]*rhs[kk]
                # demand should put here, can not put in the above rhs, 
                # rhs may be wrong because it have previous stage decision variable
                pi_rhs_values[t][n][k] += -pi[-2] * demand + pi[-1] * price * demand - price*demand # put here is better because of demand
                       
   
            avg_pi = sum(np.array(pi_values[t][n])) / K
            avg_pi_rhs = sum(pi_rhs_values[t][n]) / K
                
            # recording cuts
            if t == 0 and n == 0:
                slope1_stage.append(avg_pi[0])
                intercept1_stage.append(avg_pi_rhs)
            elif t > 0:
                slopes[t-1][n].append(avg_pi[0])
                intercepts[t-1][n].append(avg_pi_rhs)   
    
    # z_lb, z_ub = compute_ub(z_values)
    # if -z <= z_ub and -z >= z_lb:
    #     print('********************************************')
    #     print('iteration ends in iter + 1 = %d' % iter)
    #     break
    iter += 1

end = time.process_time()
print('********************************************')
final_cash = -z
print('final expected cash increment is %.2f' % final_cash)
print('ordering Q in the first peiod is %.2f' % q.x)
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)


