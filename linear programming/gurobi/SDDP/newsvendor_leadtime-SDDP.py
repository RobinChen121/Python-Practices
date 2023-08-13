# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 15:44:23 2023

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.11


Description: SDDP to solve newsvendor with lead time
    
"""

from gurobipy import *
import time
import itertools
import random

import sys 
sys.path.append("..") 
from tree import generate_sample, get_tree_strcture, generate_scenario_samples



ini_I = 0
vari_cost = 1
unit_back_cost = 10
unit_hold_cost = 2
mean_demands = [10, 10]
T = len(mean_demands)
sample_nums = [2 for t in range(T)]

trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = 1
for i in sample_nums:
    scenario_numTotal *= i

# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    sample_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
sample_detail = [[5, 15], [5, 15]]
# scenarios_full = list(itertools.product(*sample_detail)) 


iter = 0
iter_num = 5
N = 4 # sampled number of scenarios for forward computing

theta_iniValue = 0 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = theta_iniValue*T, vtype = GRB.CONTINUOUS, name = 'theta_2')
m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)

# cuts
slope1_stage = [0 for i in range(iter_num)]
intercept1_stage = [0 for i in range(iter_num)]
slopes1 = [[[ 0 for n in range(N)] for t in range(T-1)] for i in range(iter_num)]
slopes2 = [[[0 for n in range(N)] for t in range(T-1)] for i in range(iter_num)]
intercepts = [[[0 for n in range(N)] for t in range(T-1)] for i in range(iter_num)]
q_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)]


start = time.process_time()
pi_Iflow = [[[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)] for i in range(iter_num)]
pi_q = [[[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)] for i in range(iter_num)]
while iter < iter_num:  
    
    # sample a numer of scenarios from the full scenario tree
    # random.seed(10000)
    sample_scenarios = generate_scenario_samples(N, trunQuantile, mean_demands)
    sample_scenarios = [[5, 5], [5, 15], [15, 5], [15, 15]]
    sample_scenarios.sort() # sort to make same numbers together
    
    # forward
    if iter > 0:
        m.addConstr(theta >= slope1_stage[iter-1]*q + intercept1_stage[iter-1])
    m.optimize()
    if iter > 0:
        m.write('iter' + str(iter) + '_main.lp')
        pass
    # m.write('iter' + str(iter) + '_main.sol')
    
    q_values[iter][0] = [q.x for n in range(N)]
    z = m.objVal
    
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    I_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    # B is the quantity of lost sale
    B_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[t][n].addVar(lb = -theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]

    q_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    # theta_forward_values = [[0 for n in range(N)] for t in range(T)]
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]
            
            # put those cuts in the front
            if iter > 0 and t < T - 1:
                for i in range(iter):
                    for nn in range(N): # N
                        if t < T - 2:
                            m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[i][t][nn]*(I_forward[t][n]- B_forward[t][n]) + slopes2[i][t][nn]*q_forward[t][n] + intercepts[i][t][nn])
                        else:
                            m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[i][t][nn]*(I_forward[t][n]- B_forward[t][n]) + intercepts[i][t][nn]) 
                           
            if t == T - 1:                   
                m_forward[t][n].setObjective(unit_hold_cost*I_forward[t][n] + unit_back_cost*B_forward[t][n], GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(vari_cost*q_forward[t][n] + unit_hold_cost*I_forward[t][n] + unit_back_cost*B_forward[t][n] + theta_forward[t][n], GRB.MINIMIZE)
            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I  - demand)
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] - B_forward_values[t-1][n] + q_values[iter][t-1][n] - demand)
            
            # optimize
            m_forward[t][n].optimize()
            # if iter > 2:
            #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-.lp')
            #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-.sol')
            #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-.dlp')
            #     pass
            
            I_forward_values[t][n] = I_forward[t][n].x 
            B_forward_values[t][n] = B_forward[t][n].x      
            if t < T - 1:
                q_forward_values[t][n] = q_forward[t][n].x
                # theta_forward_values[t][n] = theta_forward[t][n]
            # m_forward[t][n].dispose()
    
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
    pi_values2 = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
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
                        for nn in range(N): # N
                            if t < T - 2:
                                m_backward[t][n][k].addConstr(theta_backward[t][n][k] >= slopes1[i][t][nn]*(I_backward[t][n][k]- B_backward[t][n][k]) + slopes2[i][t][nn]*q_backward[t][n][k] + intercepts[i][t][nn])
                            else:
                                m_backward[t][n][k].addConstr(theta_backward[t][n][k] >= slopes1[i][t][nn]*(I_backward[t][n][k]- B_backward[t][n][k]) + intercepts[i][t][nn]) 
            
                if t == T - 1:                   
                    m_backward[t][n][k].setObjective(unit_hold_cost*I_backward[t][n][k] + unit_back_cost*B_backward[t][n][k], GRB.MINIMIZE)
                else:
                    m_backward[t][n][k].setObjective(vari_cost*q_backward[t][n][k] + unit_hold_cost*I_backward[t][n][k] + unit_back_cost*B_backward[t][n][k] + theta_backward[t][n][k], GRB.MINIMIZE)
                if t == 0:   
                    m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == ini_I - demand)
                else:
                    m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == I_forward_values[t-1][n] - B_forward_values[t-1][n] + q_values[iter][t-1][n] - demand)
                    
                # optimize
                m_backward[t][n][k].optimize()                
                if  iter > 2 and t == 0:
                    m_backward[t][n][k].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(k+1) +'-back.lp')
                    m_backward[t][n][k].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(k+1) +'-back.sol')
                    m_backward[t][n][k].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(k+1) +'-back.dlp')
                    pass
                
                pi = m_backward[t][n][k].getAttr(GRB.Attr.Pi)
                rhs = m_backward[t][n][k].getAttr(GRB.Attr.RHS)
                
                if t < T - 1:
                    num_con = len(pi)
                    pi_expect = sum(pi_Iflow[iter-1][t+1][n])/K
                    piq_expect = sum(pi_q[iter-1][t+1][n])/K
                    for kk in range(num_con-1):
                        pi_rhs_values[t][n][k] += pi[kk]*(rhs[kk]) # - piq_expect
                    pi_rhs_values[t][n][k] += -pi[-1]*demand
                else:
                    pi_rhs_values[t][n][k] = -pi[-1] * demand + pi[-1]*q_values[iter][t-1][n]
                if t < T - 1:
                    pi_values2[t][n][k] = pi_expect * sum(pi[0:-1])
                if t > 0:
                    pi_q[iter][t][n][k] = pi[-1] * q_values[iter][t-1][n]  
                pi_Iflow[iter][t][n][k] = pi[-1]  
                                             
            
            if iter > 0 and t == 1:
                print()
            avg_pi2 = sum(pi_values2[t][n]) / K
            avg_pi1 = sum(pi_Iflow[iter][t][n]) / K
            avg_pi_rhs = sum(pi_rhs_values[t][n]) / K
            
            # recording cuts
            if t == 0 and n == 0: # n not necessary many
                slope1_stage[iter] = avg_pi2
                intercept1_stage[iter] = avg_pi_rhs
            elif t > 0:
                slopes1[iter][t-1][n] = avg_pi1
                slopes2[iter][t-1][n] = avg_pi2
                intercepts[iter][t-1][n] = avg_pi_rhs 
                if iter > 1:
                    pass                    
                
    iter += 1

end = time.process_time()
print('********************************************')
print('final expected total costs is %.2f' % z)
print('ordering Q in the first peiod is %.2f' % q_values[iter-1][0][0])
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)