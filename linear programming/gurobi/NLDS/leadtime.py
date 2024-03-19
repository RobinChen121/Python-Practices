#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:33:18 2023

@author: zhenchen

@disp:  NLDS for newsvendor lead time
    
    
"""

from gurobipy import *
import time
import itertools
import random

import sys 
sys.path.append("..") 
from tree import generate_sample, get_tree_strcture, generate_scenario_samples, getSizeOfNestedList



ini_I = 0
vari_cost = 1
unit_back_cost = 10
unit_hold_cost = 2
mean_demands = [10, 10, 10]
T = len(mean_demands)
sample_nums = [10 for t in range(T)] # sample number in one stage

trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = 1
for i in sample_nums:
    scenario_numTotal *= i

# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    sample_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
sample_detail = [[5, 15], [5, 15], [5, 15]]
# scenarios_full = list(itertools.product(*sample_detail)) 

N = 8 # sampled number of scenarios for forward computing
sample_num = N
iter = 0
iter_num = 5
sample_scenarios = generate_scenario_samples(N, trunQuantile, mean_demands)
sample_scenarios = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [15,5,5], [15,15,5], [15,5, 15], [5,15,15],[15,15,15]]
sample_scenarios.sort() # sort to make same numbers together
node_values, node_index = get_tree_strcture(sample_scenarios)
node_index.insert(0, [list(range(N))])

# number of nodes in each period
t_nodeNum = [0 for i in range(T)]
for t in range(T):
    t_nodeNum[t] = getSizeOfNestedList(node_values[t])
    
theta_iniValue = 0 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = theta_iniValue*T, vtype = GRB.CONTINUOUS, name = 'theta_2')
m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)


I_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)]
B_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)]
q_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
qpre_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
slope_1stage = [0 for iter in range(iter_num)]
intercept_1stage = [0 for iter in range(iter_num)]
slopes1 = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
slopes2 = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
intercepts = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 

start = time.process_time()
pi = [[[ [] for n in range(N)] for t in range(T)] for i in range(iter_num)]
rhs = [[[ [] for n in range(N)] for t in range(T)] for i in range(iter_num)]

while iter < iter_num:  
        
    # forward and backward
    if iter > 0:
        m.addConstr(theta >= slope_1stage[iter-1]*q + intercept_1stage[iter-1])
    m.optimize()
    # if iter >= 1:
    #     m.write('iter' + str(iter) + '_main.lp')    
    # m.write('iter' + str(iter) + '_main.sol')
    # pass
    
    q_values[iter][t] = [q.x for n in range(N)]
    z = m.objVal
    
    m_sub = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_sub[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    q_pre = [[m_sub[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'qpre_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    I = [[m_sub[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    # B is the quantity of lost sale
    B = [[m_sub[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_sub[t][n].addVar(lb = -theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]

    for t in range(T): 
        for n in range(N): 
            demand = sample_scenarios[n][t]
            
            if t == T - 1:                   
                m_sub[t][n].setObjective(unit_hold_cost*I[t][n] + unit_back_cost*B[t][n], GRB.MINIMIZE)
            else:
                m_sub[t][n].setObjective(vari_cost*q_forward[t][n] + unit_hold_cost*I[t][n] + unit_back_cost*B[t][n] + theta_forward[t][n], GRB.MINIMIZE)                     
            if t == 0:   
                m_sub[t][n].addConstr(I[t][n] - B[t][n] == ini_I  - demand)
            else:
                m_sub[t][n].addConstr(I[t][n] - B[t][n] == I_values[iter][t-1][n] - B_values[iter][t-1][n] + qpre_values[iter][t-1][n] - demand)
            if t < T - 1:
                m_sub[t][n].addConstr(q_pre[t][n] == q_values[iter][t][n])
                
            # add cut
            if t < T - 1:
                for i in range(iter):
                    m_sub[t][n].addConstr(theta_forward[t][n] >= slopes1[i][t][n]*q_forward[t][n] + slopes2[i][t][n]*(I[t][n] - B[t][n] + q_pre[t][n]) + intercepts[i][t][n])                
               
            
            
            # optimize
            m_sub[t][n].optimize()
            if iter == 2 and t == 1:
                m_sub[t][n].write('iter' + str(iter) + '_sub_' + str(t) + '^' + str(n) + '.lp')
                pass
            #     m_sub[t][n].write('iter' + str(iter) + '_sub_' + str(t) + '^' + str(n) + '.sol')
            #     # m_sub[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-.dlp')
            #     pass
            
            I_values[iter][t][n] = I[t][n].x 
            B_values[iter][t][n] = B[t][n].x      
            if t < T - 1:
                q_values[iter][t+1][n] = q_forward[t][n].x
                qpre_values[iter][t][n] = q_pre[t][n].x
            
            
            pi[iter][t][n] = m_sub[t][n].getAttr(GRB.Attr.Pi)
            rhs[iter][t][n] = m_sub[t][n].getAttr(GRB.Attr.RHS)
            
    # get cuts
    for t in range(T-1, -1, -1):
        for indice in node_index[t]:
            intercept_sum = 0
            slope1_sum = 0
            slope2_sum = 0
            for n in indice:
                demand = sample_scenarios[n][t]              
                num_con = len(pi[iter][t][n])
                if iter == 0 and t == T - 1: 
                    pass
                for kk in range(2, num_con):  # actually iter > 0 and t< T-1, below for cuts in iter > 1
                    intercept_sum += pi[iter][t][n][kk]*rhs[iter][t][n][kk]
                    # col = m_sub[t][n].getConstrs()[kk]
                    # row = m_sub[t][n].getRow(col)
                    # a = m_sub[t][n].getCol(q_pre[t][n]).getCoeff(kk)
                if iter > 0 and t < T - 1:
                    slope1_sum += pi[iter][t][n][1]
                if t > 0:
                    intercept_sum += pi[iter][t][n][0] * (-demand) 
                else:
                    intercept_sum += pi[iter][t][n][0] * (ini_I - demand)
                slope2_sum += pi[iter][t][n][0]   
                if t == T - 1: # consider T - 1 separately
                    pass
            avg_intercept = intercept_sum / len(indice)
            avg_slope1 = slope1_sum / len(indice)
            avg_slope2 = slope2_sum / len(indice)
            if t == 0:
                slope_1stage[iter] = avg_slope1
                intercept_1stage[iter] = avg_intercept
            else:
                for n in indice:
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
    
    