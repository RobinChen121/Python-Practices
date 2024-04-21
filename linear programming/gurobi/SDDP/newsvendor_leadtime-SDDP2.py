# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:44:23 2023

@author: zhen chen
@Email: chen.zhen5526@gmail.com

Python version: 3.11


Description: SDDP to solve newsvendor with lead time;
price newsvendor; 

* longer iterations seems more important than longer N;
* variable theta should be free in the initialization and set its lower bound in the constraints,
or else affecting its dual value;
* it is better to have another constratint B <= d to avoid multiple solutions in the LPs and affect the cuts (not necessary);


for case:
ini_I = 0
ini_cash = 0
vari_cost = 1
price = 10
unit_hold_cost = 0
unit_salvage = 0
mean_demands = [10, 10, 10] 
T = len(mean_demands)
sample_nums = [10 for t in range(T)] # sample number in one stage

iter = 0
iter_num = 21
N = 10 # sampled number of scenarios for forward computing

SDP cost is 171.8, and Q*_1 is 24;
SDDP cost is 170.9 and Q*_1 is 17, python cpu time is about 62s.
  
"""

from gurobipy import *
import time
import itertools
import random

import sys 
sys.path.append("..") 
from tree import generate_sample, get_tree_strcture, generate_scenario_samples



ini_I = 0
ini_cash = 0
vari_cost = 1
price = 10
unit_hold_cost = 0
unit_salvage = 0.5
mean_demands = [10, 10, 10] 
T = len(mean_demands)
sample_nums = [10 for t in range(T)] # sample number in one stage
overhead_cost = [0 for t in range(T)]


iter = 0
iter_num = 21
N = 8 # sampled number of scenarios for forward computing



trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = 1
for i in sample_nums:
    scenario_numTotal *= i

# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    sample_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
sample_detail = [[5, 15], [5, 15], [5, 15]]




theta_iniValue = -500 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_2')
m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))

# cuts
slope1_stage = [0 for i in range(iter_num)]
slope2_stage = [0 for i in range(iter_num)]
slope3_stage = [0 for i in range(iter_num)]
intercept1_stage = [0 for i in range(iter_num)]
slopes1 = [[[ 0 for n in range(N)] for t in range(T-1)] for iter in range(iter_num)] 
slopes2 = [[[ 0 for n in range(N)] for t in range(T-1)] for iter in range(iter_num)] 
slopes3 = [[[ 0 for n in range(N)] for t in range(T-1)] for iter in range(iter_num)] 
intercepts = [[[ 0 for n in range(N)] for t in range(T-1)] for iter in range(iter_num)] 
q_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
qpre_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 

start = time.process_time()
pi = [[[ [] for n in range(N)] for t in range(T)] for i in range(iter_num)]
rhs = [[[ [] for n in range(N)] for t in range(T)] for i in range(iter_num)]

pi_d = [[[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] for i in range(iter_num)]
pi_q = [[[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] for i in range(iter_num)]

while iter < iter_num:  
    
    # sample a numer of scenarios from the full scenario tree
    # random.seed(10000)
    sample_scenarios = generate_scenario_samples(N, trunQuantile, mean_demands)
    sample_scenarios = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [15,5,5], [15,15,5], [15,5, 15], [5,15,15],[15,15,15]]
    sample_scenarios.sort() # sort to make same numbers together
    
    # forward
    if iter > 0:
        m.addConstr(theta >= slope2_stage[iter-1]*(-vari_cost*q) + slope3_stage[iter-1]*q + intercept1_stage[iter-1])
    m.optimize()
    if iter == 3:
        m.write('iter' + str(iter) + '_main.lp')
        m.write('iter' + str(iter) + '_main.sol')
        pass
    
    q_values[iter][0] = [q.x for n in range(N)]
    z = m.objVal
    
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    q_pre_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'qpre_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    
    I_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]    
    B_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)] # B is the quantity of lost sale
    theta_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    cash_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1)+ '^' + str(n+1)) for n in range(N)] for t in range(T)]
    
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    # theta_forward_values = [[0 for n in range(N)] for t in range(T)]
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]
            
            if t == T - 1:                   
                m_forward[t][n].setObjective(-price*(demand - B_forward[t][n]) - unit_salvage*I_forward[t][n] + overhead_cost[t], GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(vari_cost*q_forward[t][n] -price*(demand - B_forward[t][n]) + overhead_cost[t] + theta_forward[t][n], GRB.MINIMIZE)                     
            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I  - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] + price*B_forward[t][n] == ini_cash - vari_cost*q_values[iter][t][n] + price*demand - overhead_cost[t])
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] + qpre_values[iter][t-1][n] - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] + price*B_forward[t][n] == cash_forward_values[t-1][n]- vari_cost*q_values[iter][t][n] + price*demand - overhead_cost[t])
           
            if t < T - 1:                 
                m_forward[t][n].addConstr(q_pre_forward[t][n] == q_values[iter][t][n])
                m_forward[t][n].addConstr(theta_forward[t][n] >= theta_iniValue*(T-1-t))
                              
            # add cut in the back
            if t < T - 1:
                for i in range(iter):
                    for nn in range(N):
                        # careful, some notations should be nn
                        m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[i][t][nn]*(I_forward[t][n] + q_pre_forward[t][n]) + slopes2[i][t][nn]*(cash_forward[t][n]- vari_cost*q_forward[t][n])\
                                                          + slopes3[i][t][nn]*q_forward[t][n] + intercepts[i][t][nn])               
               
            
            # optimize
            m_forward[t][n].optimize()
            # if iter == 1 and t == 0 and n == 0:
            #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '.lp')
            #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '.sol')
            #     pass
                # m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-.dlp')
            
            I_forward_values[t][n] = I_forward[t][n].x 
            B_forward_values[t][n] = B_forward[t][n].x   
            cash_forward_values[t][n] = cash_forward[t][n].x  
            if iter == 2 and t == 1:
                pass
            if t < T - 1:
                q_values[iter][t+1][n] = q_forward[t][n].x
                qpre_values[iter][t][n] = q_pre_forward[t][n].x
            # m_forward[t][n].dispose()
    
    # backward
    m_backward = [[[Model() for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    q_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    q_pre_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'qpre_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T)] 
    I_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T)]
    # B is the quantity of lost sale
    B_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward = [[[m_backward[t][n][s].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for s in range(sample_nums[t])] for n in range(N)] for t in range(T - 1)]
    cash_backward = [[[m_backward[t][n][s].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1)+ '^' + str(n+1)) for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    
    cash_backward_values = [[0 for n in range(N)] for t in range(T)]
    intercept_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    slope1_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
    slope2_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
    slope3_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
    
    # it is better t in the first loop
    for t in range(T - 1, -1, -1):
       for n in range(N):          
            S = len(sample_detail[t])
            for s in range(S):
                demand = sample_detail[t][s]                           
                if t == T - 1:                   
                    m_backward[t][n][s].setObjective(-price*(demand - B_backward[t][n][s]) - unit_salvage*I_backward[t][n][s] + overhead_cost[t], GRB.MINIMIZE)
                else:
                    m_backward[t][n][s].setObjective(vari_cost*q_backward[t][n][s] -price*(demand - B_backward[t][n][s])  + overhead_cost[t] + theta_backward[t][n][s], GRB.MINIMIZE)
                if t == 0:   
                    m_backward[t][n][s].addConstr(I_backward[t][n][s] - B_backward[t][n][s] == ini_I - demand)
                    m_backward[t][n][s].addConstr(cash_backward[t][n][s] + price*B_backward[t][n][s] == ini_cash - vari_cost*q_values[iter][t][n] + price*demand- overhead_cost[t])
                else:
                    m_backward[t][n][s].addConstr(I_backward[t][n][s] - B_backward[t][n][s] == I_forward_values[t-1][n] + qpre_values[iter][t-1][n] - demand)
                    m_backward[t][n][s].addConstr(cash_backward[t][n][s] + price*B_backward[t][n][s] == cash_forward_values[t-1][n]- vari_cost*q_values[iter][t][n] + price*demand- overhead_cost[t])
                
                if t < T - 1:                   
                    m_backward[t][n][s].addConstr(q_pre_backward[t][n][s] == q_values[iter][t][n])
                    m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= theta_iniValue*(T-1-t))
                                       
                # put those cuts in the back
                if iter > 0 and t < T - 1:
                    for i in range(iter):
                        for nn in range(N): # N
                            m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= slopes1[i][t][nn]*(I_backward[t][n][s] + q_pre_backward[t][n][s]) +\
                                                          slopes2[i][t][nn]*(cash_backward[t][n][s]- vari_cost*q_backward[t][n][s])\
                                                              + slopes3[i][t][nn]*q_backward[t][n][s] + intercepts[i][t][nn])
               
                # optimize
                m_backward[t][n][s].optimize()   
                pi = m_backward[t][n][s].getAttr(GRB.Attr.Pi)
                rhs = m_backward[t][n][s].getAttr(GRB.Attr.RHS)
                
                # if  iter == 2 and t == 2:
                #     m_backward[t][n][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(s+1) +'-back.lp')
                #     m_backward[t][n][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(s+1) +'-back.sol')  
                #     pass
                
                #     m_backward[t][n][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(s+1) +'-back.dlp')
                               
                num_con = len(pi)
                intercept_sum = 0
                for kk in range(3, num_con):  # when t=T-1, num_con < 2
                    intercept_sum += pi[kk]*rhs[kk]
                if t > 0:
                    intercept_sum += pi[0] * (-demand) 
                    intercept_sum += pi[1]*price*demand - price*demand
                else:
                    intercept_sum += pi[0] * (ini_I - demand)
                    intercept_sum += pi[1]*ini_cash + pi[1]*price*demand - price*demand - pi[1]*overhead_cost[t]
                intercept_sum += overhead_cost[t]
                slope1_values[t][n][s] = pi[0]                                                          
                slope2_values[t][n][s] = pi[1]
                if t < T - 1:
                    slope3_values[t][n][s] = pi[2]
                intercept_values[t][n][s] = intercept_sum

            
            avg_intercept = sum(intercept_values[t][n]) / S
            avg_slope1 = sum(slope1_values[t][n]) / S
            avg_slope2 = sum(slope2_values[t][n]) / S
            avg_slope3 = sum(slope3_values[t][n]) / S
            if t == 0:
                slope1_stage[iter] = avg_slope1 
                slope2_stage[iter] = avg_slope2
                slope3_stage[iter] = avg_slope3
                intercept1_stage[iter] = avg_intercept
            else:
                slopes1[iter][t-1][n] = avg_slope1 # a scaler
                slopes2[iter][t-1][n] = avg_slope2 
                slopes3[iter][t-1][n] = avg_slope3 
                intercepts[iter][t-1][n] = avg_intercept  
                
    iter += 1

end = time.process_time()
print('********************************************')
print('final expected total costs is %.2f' % -z)
print('ordering Q in the first peiod is %.2f' % q_values[iter-1][0][0])
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)