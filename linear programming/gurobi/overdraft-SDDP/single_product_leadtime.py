#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:18:39 2024

@author: zhenchen

@disp:  business overdraft for lead time in single product problem;

longer iterations seems more important than longer N;
* should not have interest free amount since this would cause the solver obtain the wrong values for Ws when initial cash is larger than overhead cost;


ini_I = 0
ini_cash = 0
vari_cost = 1
price = 10
unit_back_cost = 0
unit_hold_cost = 0
unit_salvage = 0
mean_demands = [20, 35, 20]
T = len(mean_demands)
sample_nums = [10 for t in range(T)]
overhead_cost = [100 for t in range(T)]

r0 = 0.01
r1 = 0.1
r2 = 1 # penalty interest rate for overdraft exceeding the limit
U = 1000 # overdraft limit
iter_num = 15
N = 15 # sampled number of scenarios for forward computing


SDP optimal is 133, q=46;    
SDDP optimal is 134, q=43, cpu time is 73s;

******************************
ini_I = 0
ini_cash = 0
vari_cost = 1
price = 10
unit_back_cost = 0
unit_hold_cost = 0
unit_salvage = 0.5
mean_demands = [20, 10, 20, 10]
T = len(mean_demands)
sample_nums = [10 for t in range(T)]
overhead_cost = [50 for t in range(T)]

r0 = 0
r1 = 0.1
r2 = 2 # penalty interest rate for overdraft exceeding the limit
U = 500 # overdraft limit
iter_num = 15
N = 10 # sampled number of scenarios for forward computing

SDP optimal result is 118.77; 
SDDP is 127.23, time 11.87s; (N=N, iter_num=15) # the first N is the number of cuts added in the LPs
SDDP is 122.45, time 36.40s;(N=N, iter_num=27)
SDDP is 122.47, time 36.40s;(N=N, iter_num=35)
SDDP with confidence interval is 134.70, time 14.94s;
SDDP is 121.15, time 20.55s; (N=N, iter_num=20)
SDDP is 116.87, time 4.09s; (N=1, iter_num=20)
SDDP is 137.89, time is 49.04s; (N=1, iter_num=15)    

******************************
mean_demands = [10, 10, 10, 10]
SDP optimal result is 26.68;
mean_demands = [15, 15, 15, 15]
SDP optimal result is 167.31, java running time is 39s;
for 4 periods [10, 20, 10, 20], solution 215.48, python running more than 4 hours and can't get a solution, while java 31s; 

the lower and upper bound seems to be affected by the iter limit or scenario number much;

"""

from gurobipy import *
import itertools
import random
import time
import numpy as np
import os
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from tree import *
from write_to_file import write_to_csv



ini_I = 0
ini_cash = 0
vari_cost = 1
price = 10
unit_back_cost = 0
unit_hold_cost = 0
unit_salvage = 0.5
mean_demands = [15, 15, 15, 15] # [10, 20, 10, 20]
T = len(mean_demands)
if T == 4:
    opt = 167.31 # 215.48 # 
else:
    opt = 26.68
 
sample_num = 10 # change 1
sample_nums = [sample_num for t in range(T)] 
overhead_cost = [50 for t in range(T)]

r0 = 0
r1 = 0.1
r2 = 2 # penalty interest rate for overdraft exceeding the limit
U = 500 # overdraft limit
iter_limit = 30
time_limit = 120 # time limit
N = 10 # sampled number of scenarios for forward computing    # change 2
cut_select_num = N

trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = 1
for i in sample_nums:
    scenario_numTotal *= i

    
# detailed samples in each period
sample_details = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    sample_details[t] = generate_samples(sample_nums[t], trunQuantile, mean_demands[t])
# sample_details = [[5, 15], [5, 15], [5, 15]]  # change 3
scenarios_full = list(itertools.product(*sample_details)) 

iter = 0
theta_iniValue = -500 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
W0 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^0')
W1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^1')
W2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^2')
theta = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_2')
m.setObjective(overhead_cost[0] + vari_cost*q  + r2*W2 + r1*W1 - r0*W0 + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))
m.addConstr(W1 <= U)
m.addConstr(-vari_cost*q - W0 + W1 + W2 == overhead_cost[0] - ini_cash)
theta_value = 0 

# cuts
slope1_stage = []
intercept1_stage = []
slopes1 = []
slopes2 = []
slopes3 = []
intercepts = []
q_values = []
qpre_values = [] 
W0_values = []
W1_values = []
W2_values = []

cpu_time = 0
start = time.process_time()
while iter < iter_limit:  
    
    slopes1.append([[ 0 for n in range(N)] for t in range(T)])
    slopes2.append([[0 for n in range(N)] for t in range(T)])
    slopes3.append([[0 for n in range(N)] for t in range(T)])
    intercepts.append([[0 for n in range(N)] for t in range(T-1)])
    q_values.append([[0 for n in range(N)] for t in range(T)])
    qpre_values.append([[0 for n in range(N)] for t in range(T)])   
    
    # sample a numer of scenarios from the full scenario tree
    #  random.seed(10000)
    sample_scenarios = generate_scenarios(N, sample_num, sample_details)
    # sample_scenarios = generate_scenarios2(N, trunQuantile, mean_demands)
    # sample_scenarios = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [15,5,5], [15,15,5], [15,5, 15], [5,15,15],[15,15,15]] # change 4
    sample_scenarios.sort() # sort to make same numbers together
    
    # forward
    if iter > 0:        
        m.addConstr(theta >= slope1_stage[-1][0]*(ini_I) + slope1_stage[-1][1]*(ini_cash-vari_cost*q) + slope1_stage[-1][2]*q + intercept1_stage[-1])        
        m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    
    q_values[-1][0] = [q.x for n in range(N)]  
    if iter == 14:
        m.write('iter' + str(iter+1) + '_main.lp')    
        m.write('iter' + str(iter+1) + '_main.sol')
        pass
    
    W0_values.append(W0.x)
    W1_values.append(W1.x)
    W2_values.append(W2.x)
    theta_value = theta.x
    z = m.objVal    
    z_values = [[0 for t in range(T+1)] for n in range(N)] # for computing confidence interval
    for n in range(N):
        z_values[n][0] = m.objVal - theta.x
    
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    q_pre_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'qpre_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T-1)]
    I_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    cash_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1)+ '^' + str(n+1)) for n in range(N)] for t in range(T)]
    W0_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W1_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W2_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    
    # B is the quantity of lost sale
    B_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
 
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    W0_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    W1_forward_values = [[0 for n in range(N)] for t in range(T-1)]
    W2_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]
            
            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] + price*B_forward[t][n] == ini_cash - overhead_cost[t] - vari_cost*q_values[-1][t][n]\
                                          -r1*W1_values[-1] + r0*W0_values[-1]\
                                              -r2*W2_values[-1] + price*demand)
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] + qpre_values[-1][t-1][n] - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] + price*B_forward[t][n] == cash_forward_values[t-1][n] - overhead_cost[t] \
                                                 - vari_cost*q_values[-1][t][n]\
                                                     -r1*W1_forward_values[t-1][n] + r0*W0_forward_values[t-1][n]\
                                                         -r2*W2_values[-1] + price*demand)
             
            if t < T - 1:
                m_forward[t][n].addConstr(q_pre_forward[t][n] == q_values[-1][t][n]) 
            # m_forward[t][n].addConstr(B_forward[t][n] <= demand)           
            if t == T - 1:                   
                m_forward[t][n].setObjective(-price*(demand - B_forward[t][n]) - unit_salvage*I_forward[t][n], GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(overhead_cost[t] + vari_cost*q_forward[t][n] - price*(demand - B_forward[t][n])\
                                             + r2*W2_forward[t][n]\
                                             + r1*W1_forward[t][n] - r0*W0_forward[t][n] + theta_forward[t][n], GRB.MINIMIZE)  
                
                m_forward[t][n].addConstr(W1_forward[t][n] <= U) 
                m_forward[t][n].addConstr(cash_forward[t][n] - vari_cost*q_forward[t][n] - W0_forward[t][n]\
                                          + W1_forward[t][n] + W2_forward[t][n] == overhead_cost[t+1])
                   
                m_forward[t][n].addConstr(theta_forward[t][n] >= theta_iniValue*(T-1-t))                  
            
            # put those cuts in the back
            if iter > 0 and t < T - 1:
                for i in range(iter):
                    for nn in range(cut_select_num): # N
                        m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[i][t][nn]*(I_forward[t][n]+ q_pre_forward[t][n])\
                                                  + slopes3[i][t][nn]*q_forward[t][n]\
                                                  + slopes2[i][t][nn]*(cash_forward[t][n]- vari_cost*q_forward[t][n]-r2*W2_forward[t][n]\
                                             - r1*W1_forward[t][n]+r0*W0_forward[t][n])\
                                                + intercepts[i][t][nn])
            
            # optimize
            m_forward[t][n].Params.LogToConsole = 0
            m_forward[t][n].optimize()
            I_forward_values[t][n] = I_forward[t][n].x 
            
            if t < T - 1: # for computing cnofidence interval
                z_values[n][t+1] = m_forward[t][n].objVal - theta_forward[t][n].x
            else:
                z_values[n][t+1] = m_forward[t][n].objVal
                           
            B_forward_values[t][n] = B_forward[t][n].x  
            cash_forward_values[t][n] = cash_forward[t][n].x 
            # if iter == 1 and t == 0:
            #     m_forward[t][n].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '.lp') 
            #     m_forward[t][n].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '.sol') 
            #     pass
            if t < T - 1:
                q_values[-1][t+1][n] = q_forward[t][n].x
                qpre_values[-1][t][n] = q_pre_forward[t][n].x
                W1_forward_values[t][n] = W1_forward[t][n].x
                W0_forward_values[t][n] = W0_forward[t][n].x
                W2_forward_values[t][n] = W2_forward[t][n].x
                
    # backward
    m_backward = [[[Model() for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    q_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    q_pre_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'qpre_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T-1)] 
    
    I_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T)]
    cash_backward = [[[m_backward[t][n][s].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T)]    
    W0_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    W1_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    W2_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    
    # B is the quantity of lost sale
    B_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward = [[[m_backward[t][n][s].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for s in range(sample_nums[t])] for n in range(N)] for t in range(T - 1)]
    
    intercept_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    slope1_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
    slope2_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
    slope3_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
    
    for t in range(T-1, -1, -1):    
        for n in range(N):      
            S = len(sample_details[t])
            for s in range(S):
                demand = sample_details[t][s]

                if t == T - 1:                   
                    m_backward[t][n][s].setObjective(-price*(demand - B_backward[t][n][s]) - unit_salvage*I_backward[t][n][s], GRB.MINIMIZE)
                else:
                    m_backward[t][n][s].setObjective(overhead_cost[t] + vari_cost*q_backward[t][n][s] - price*(demand - B_backward[t][n][s])\
                                                     + r2*W2_backward[t][n][s]
                                                     + r1*W1_backward[t][n][s] - r0*W0_backward[t][n][s] + theta_backward[t][n][s], GRB.MINIMIZE)  
                if t == 0:   
                    m_backward[t][n][s].addConstr(I_backward[t][n][s] - B_backward[t][n][s] == ini_I - demand)
                    m_backward[t][n][s].addConstr(cash_backward[t][n][s]  + price*B_backward[t][n][s] == ini_cash - overhead_cost[t] - vari_cost*q_values[-1][t][n]\
                                                  - r2*W2_values[-1] - r1*W1_values[-1] + r0*W0_values[-1] + price*demand)
                else:
                    m_backward[t][n][s].addConstr(I_backward[t][n][s] - B_backward[t][n][s] == I_forward_values[t-1][n] + qpre_values[-1][t-1][n] - demand)
                    m_backward[t][n][s].addConstr(cash_backward[t][n][s] + price*B_backward[t][n][s] == cash_forward_values[t-1][n]- overhead_cost[t]\
                                                  - vari_cost*q_values[-1][t][n]\
                                                 - r2*W2_forward_values[t-1][n]- r1*W1_forward_values[t-1][n]\
                                                  + r0*W0_forward_values[t-1][n] + price*demand)
            
                if t < T - 1:
                    m_backward[t][n][s].addConstr(q_pre_backward[t][n][s] == q_values[-1][t][n]) 
                # m_backward[t][n][s].addConstr(B_backward[t][n][s] <= demand)
                if t < T - 1:                   
                    m_backward[t][n][s].addConstr(W1_backward[t][n][s] <= U)
                    m_backward[t][n][s].addConstr(cash_backward[t][n][s] - vari_cost*q_backward[t][n][s] - W0_backward[t][n][s]\
                                                  + W1_backward[t][n][s] + W2_backward[t][n][s] == overhead_cost[t+1])
                    m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= theta_iniValue*(T-1-t))
                            
                # put those cuts in the back
                if iter > 0 and t < T - 1:
                    for i in range(iter):
                        for nn in range(cut_select_num): # N
                            m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= slopes1[i][t][nn]*(I_backward[t][n][s]+ q_pre_backward[t][n][s])\
                                                      + slopes3[i][t][nn]*q_backward[t][n][s]\
                                                      + slopes2[i][t][nn]*(cash_backward[t][n][s]- vari_cost*q_backward[t][n][s] -r2*W2_backward[t][n][s]\
                                                 - r1*W1_backward[t][n][s]+r0*W0_backward[t][n][s])\
                                                    + intercepts[i][t][nn])
                               
                # optimize
                m_backward[t][n][s].Params.LogToConsole = 0
                m_backward[t][n][s].optimize()
                
                if t < T - 1:
                    temp = W2_backward[t][n][s].X
                    if temp > 0:
                        pass
                    
                
                pi = m_backward[t][n][s].getAttr(GRB.Attr.Pi)
                rhs = m_backward[t][n][s].getAttr(GRB.Attr.RHS)  
                               
                num_con = len(pi)
                if t < T - 1:
                    # important
                    intercept_values[t][n][s] += -pi[0]*demand + pi[1]*price*demand - pi[1]*overhead_cost[t] - price*demand + overhead_cost[t+1] # + pi[3]*U + pi[4]*overhead_cost[t+1]-pi[5]*theta_iniValue*(T-1-t) 
                else:
                    intercept_values[t][n][s] += -pi[0]*demand + pi[1]*price*demand - pi[1]*overhead_cost[t] - price*demand 
                for sk in range(3, num_con):
                    intercept_values[t][n][s] += pi[sk]*rhs[sk]
                
                slope1_values[t][n][s] = pi[0]                                                         
                slope2_values[t][n][s] = pi[1]
                if t < T -1:
                    slope3_values[t][n][s] = pi[2]
                # if iter == 3 and t == 2 and n == 0:
                #     a_test = intercept_values[t][n][s]
                #     m_backward[t][n][s].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '-' + str(s+1) +'back.lp')
                #     m_backward[t][n][s].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '-' + str(s+1) +'back.sol')
                #     filename = 'iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '-' + str(s+1) + '.txt'
                #     with open(filename, 'w') as f:
                #         f.write('demand=' +str(demand)+'\n')
                #         f.write(str(pi)+'\n')  
                #         f.write(str(rhs))
                #     pass
            
            avg_intercept = sum(intercept_values[t][n]) / S
            avg_slope1 = sum(slope1_values[t][n]) / S
            avg_slope2 = sum(slope2_values[t][n]) / S
            avg_slope3 = sum(slope3_values[t][n]) / S
            if t == 0:
                if n == 0:
                    temp = [avg_slope1, avg_slope2, avg_slope3]
                    slope1_stage.append(temp)
                    intercept1_stage.append(avg_intercept)
                    if iter == 0:
                        pass
            else:
                slopes1[-1][t-1][n] = avg_slope1 
                slopes2[-1][t-1][n] = avg_slope2  
                slopes3[-1][t-1][n] = avg_slope3
                intercepts[-1][t-1][n] = avg_intercept 
                if iter == 3 and t == 2:
                    pass
             
             
                
    z_lb, z_ub, z_mean = compute_ub(z_values) # for computing confidence interval
    # if -z <= z_ub and -z >= z_lb:
    #     print('********************************************')
    #     print('iteration ends in iter + 1 = %d' % iter)
    #     break
    iter += 1
    end = time.process_time()
    cpu_time = end - start

print('********************************************')
final_value = -z
Q1 = q_values[iter-1][0][0]
print('after %d iteration: ' % iter)
print('final expected total costs is %.2f' % final_value)
print('ordering Q in the first peiod is %.2f' % Q1)
print('cpu time is %.3f s' % cpu_time)        
gap = (-opt + final_value)/opt               
print('optimaility gap is %.2f%% s' % (100*gap))  
z_lb, z_ub, z_mean = compute_ub(z_values) # for computing confidence interval
lb = -np.mean(np.sum(z_values, axis=1))
print('expected lower bound gap is %.2f' % lb)  
gap2 = abs((z+lb)/z)
print('lower bound and upper bound gap is %.2f%%' % (100*gap2))  
print('confidence interval for expected objective is [%.2f,  %.2f]' % (-z_ub, -z_lb))  
        
