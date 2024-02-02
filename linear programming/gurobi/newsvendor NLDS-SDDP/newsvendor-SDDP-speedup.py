#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 12:17:04 2024

@author: zhenchen

@disp:  
    speedup is a strategy to select the last cut_num = 600 cuts;
    this method seems performs poor;

-----
ini_I = 0
vari_cost = 1
unit_bacs_cost = 10
unit_hold_cost = 2
mean_demands = [10, 20, 10, 20, 10, 20, 10, 20]
----
218.41 for sdp optimal cost, java 0.5s;
    
223.4 for sddp dynamic cut selection with speedup(cut_num = 600), 407.79s on a mac for iter no. 21, sample no. 20;
212.7 for sddp dynamic cut selection with speedup(cut_num = 100), 327.32s on a mac for iter no. 21, sample no. 20;    
210.33 for sddp dynamic cut selection with speedup(cut_num = 50), 283.41s on a mac for iter no. 21, sample no. 20;    
67.35  for sddp dynamic cut selection with speedup(cut_num = 20), 251.72s on a mac for iter no. 21, sample no. 20;    

"""

from gurobipy import *
import time
import itertools
import random
import numpy as np

import sys 
sys.path.append("..") 
from tree import generate_sample, get_tree_strcture, generate_scenario_samples


cut_num = 50
ini_I = 0
vari_cost = 1
unit_bacs_cost = 10
unit_hold_cost = 2
mean_demands = [10, 20, 10, 20, 10, 20, 10, 20]
T = len(mean_demands)
sample_nums = [10 for t in range(T)]

trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = 1
for i in sample_nums:
    scenario_numTotal *= i

# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    sample_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
# samples_detail = [[5, 15], [5, 15]]
# scenarios_full = list(itertools.product(*sample_detail)) 


iter_num = 21
NN = 20 # sampled number of scenarios for forward computing

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
q_values = [0 for iter in range(iter_num)]

# kk = [1, 2, 4, 5]
kk = [NN for i in range(4)]
slopes = [[] for t in range(T)]
intercepts = [[] for t in range(T)]
start = time.process_time()
iter = 0
while iter < iter_num:  
    
    N = kk[iter] if iter < len(kk) else kk[-1] # this N is the k in the JCAM 2015 paper
    
    # sample a numer of scenarios from the full scenario tree
    # random.seed(10000)
    sample_scenarios = generate_scenario_samples(N, trunQuantile, mean_demands)
    # sample_scenarios= random.sample(scenarios_full, N) # sampling without replacement
    sample_scenarios.sort() # sort to make same numbers together
    
    # forward
    if iter > 0:
        m.addConstr(theta >= slope1_stage[-1]*q + intercept1_stage[-1])
    m.update()
    m.optimize()
    # m.write('iter' + str(iter) + '_main2.lp')    
    # m.write('iter' + str(iter) + '_main2.sol')
    
    q_values[iter] = q.x
    theta_value = theta.x
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
    theta_forward_values = [[0 for n in range(N)] for t in range(T)]
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]
            
            if t == T - 1:                   
                m_forward[t][n].setObjective(unit_hold_cost*I_forward[t][n] + unit_bacs_cost*B_forward[t][n], GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(vari_cost*q_forward[t][n] + unit_hold_cost*I_forward[t][n] + unit_bacs_cost*B_forward[t][n] + theta_forward[t][n], GRB.MINIMIZE)
            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I + q_values[iter] - demand)
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] - B_forward_values[t-1][n] + q_forward_values[t-1][n] - demand)   


            # put those cuts in the back
            if iter > 0 and t < T - 1:
                cuts_len = len(slopes[t])
                for i in range(cuts_len)[-cut_num:]:
                    m_forward[t][n].addConstr(theta_forward[t][n] >= slopes[t][i]*(I_forward[t][n]- B_forward[t][n] + q_forward[t][n]) + intercepts[t][i])
            # optimize
            m_forward[t][n].optimize()
            # m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-2.lp')                 
            I_forward_values[t][n] = I_forward[t][n].x 
            B_forward_values[t][n] = B_forward[t][n].x      
            if t < T - 1:
                q_forward_values[t][n] = q_forward[t][n].x
                theta_forward_values[t][n] = theta_forward[t][n]
                    
    
    # backward
    m_backward = [[[Model() for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    q_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    I_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for s in range(sample_nums[t])]  for n in range(N)] for t in range(T)]
    # B is the quantity of lost sale
    B_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward = [[[m_backward[t][n][s].addVar(lb = -theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for s in range(sample_nums[t])] for n in range(N)] for t in range(T - 1)]

    q_backward_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    I_backward_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    B_backward_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    pi_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    pi_rhs_values = [[[0  for s in range(sample_nums[t])] for n in range(N)] for t in range(T)] 
    
    for t in range(T - 1, -1, -1):
       for n in range(N):
            S = len(sample_detail[t])
            for s in range(S):
                demand = sample_detail[t][s]
                
                if t == T - 1:                   
                    m_backward[t][n][s].setObjective(unit_hold_cost*I_backward[t][n][s] + unit_bacs_cost*B_backward[t][n][s], GRB.MINIMIZE)
                else:
                    m_backward[t][n][s].setObjective(vari_cost*q_backward[t][n][s] + unit_hold_cost*I_backward[t][n][s] + unit_bacs_cost*B_backward[t][n][s] + theta_backward[t][n][s], GRB.MINIMIZE)
                if t == 0:   
                    m_backward[t][n][s].addConstr(I_backward[t][n][s] - B_backward[t][n][s] == ini_I + q_values[iter] - demand)
                else:
                    m_backward[t][n][s].addConstr(I_backward[t][n][s] - B_backward[t][n][s] == I_forward_values[t-1][n] - B_forward_values[t-1][n] + q_forward_values[t-1][n] - demand)
                

                 # put those cuts in the back
                if iter > 0 and t < T - 1:
                    cuts_len = len(slopes[t])
                    for i in range(cuts_len)[-cut_num:]:
                        m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= slopes[t][i]*(I_backward[t][n][s]- B_backward[t][n][s] + q_backward[t][n][s]) + intercepts[t][i])
 
                # optimize
                m_backward[t][n][s].optimize()   
                I_backward_values[t][n][s] = I_backward[t][n][s].x 
                B_backward_values[t][n][s] = B_backward[t][n][s].x      
                if t < T - 1:
                    q_backward_values[t][n][s] = q_backward[t][n][s].x
                        
                        
                # if t == 0 and n == 0 and iter > 0:
                #     m_backward[t][n][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(s+1) +'-2bacs.lp')
                # if t > 0:
                #     m_backward[t][n][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(s+1) +'-2bacs.lp')
                
                pi = m_backward[t][n][s].getAttr(GRB.Attr.Pi)
                rhs = m_backward[t][n][s].getAttr(GRB.Attr.RHS)
                if t < T - 1:
                    num_con = len(pi)
                    for ss in range(1, num_con):
                        pi_rhs_values[t][n][s] += pi[ss]*rhs[ss]
                    pi_rhs_values[t][n][s] += -pi[0]*demand 
                else:
                    pi_rhs_values[t][n][s] = -pi[0] * demand
                pi_values[t][n][s] = pi[0]
                # m_backward[t][n][s].dispose()
            

            avg_pi = sum(pi_values[t][n]) / S
            avg_pi_rhs = sum(pi_rhs_values[t][n]) / S
            
            # recording cuts
            if t == 0 and n == 0:
                slope1_stage.append(avg_pi)
                intercept1_stage.append(avg_pi_rhs)
            elif t > 0:
                slopes[t-1].append(avg_pi)
                intercepts[t-1].append(avg_pi_rhs)   
            print()
            
    iter += 1

end = time.process_time()
print('********************************************')
print('final expected total costs is %.2f' % z)
print('ordering Q in the first peiod is %.2f' % q_values[iter-1])
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)





