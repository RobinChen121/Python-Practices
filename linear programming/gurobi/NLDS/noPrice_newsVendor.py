#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 14:14:38 2023

@author: chen
@desp: use sddp to solve traditional newsvendor problem without price.

inventory flow constraint is in the last for ease of removing and updating
"""

import numpy as np
import scipy.stats as st
from gurobipy import *
import time
from functools import reduce
import itertools
import random
import time
import sys 
sys.path.append("..") 
from tree import * 


start = time.process_time()
ini_I = 0
vari_cost = 1
unit_back_cost = 10
unit_hold_cost = 2
mean_demands = [10, 10]
sample_nums = [10, 10]
T = len(mean_demands)
trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = reduce(lambda x, y: x * y, sample_nums, 1)

# samples_detail is the detailed samples in each period
samples_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    samples_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])

# samples_detail = [[5, 15], [5, 15]]
scenarios = list(itertools.product(*samples_detail)) 
sample_num = 20
samples= random.sample(scenarios, sample_num) # sampling without replacement
samples.sort() # sort to make same numbers together
node_values, node_index = get_tree_strcture(samples)

theta_iniValue = 0 # initial theta values in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = theta_iniValue*T, vtype = GRB.CONTINUOUS, name = 'theta_2')

# number of nodes in each period
t_nodeNum = [0 for i in range(T)]
for t in range(T):
    t_nodeNum[t] = getSizeOfNestedList(node_values[t])
# decision variables from stage 2 to stage T+1
m_sub = [[Model() for j in range(t_nodeNum[t])] for t in range(T)] 
q_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]
I_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
B_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
theta_sub = [[m_sub[t][j].addVar(lb = theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]


iter = 1
iter_num = 10
pi_sub_detail_values = [[[[] for s in range(t_nodeNum[t])] for t in range(T)] for iter in range(iter_num)] 
q_detail_values = [[[] for t in range(T)] for iter in range(iter_num)] 
for i in range(iter_num):
    for t in range(T):
        if t == 0:
            q_detail_values[i][t] = 0
        else:
            q_detail_values[i][t] = [0 for s in range(t_nodeNum[t-1])]

while iter <= iter_num:   
        
    # forward computation    
    # solve the first stage model    
    m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)
    m.update()
    m.optimize()
    # m.write('iter' + str(iter) + '_main.lp')    
    # m.write('iter' + str(iter) + '_main.sol')
    
    print(end = '')
    q_value = q.x
    q_detail_values[iter - 1][0] = q_value
    theta_value = theta.x
    z = m.objVal
    
    I_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    B_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    pi_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    pi_rhs_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    d_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)]
    pi_values2 = [[0 for s in range(sample_num)] for t in range(T)] 
    pi_rhs_values2 = [[0 for s in range(sample_num)] for t in range(T)] 
    
    # forward and backward  
    for t in range(T):     
        slope = [0 for j in range(t_nodeNum[t])]
        intercept = [0 for j in range(t_nodeNum[t])]
        for j in range(t_nodeNum[t]): 
            obj = [0.0 for i in range(t_nodeNum[t])] 
            index = node_index[t][j][0]
            demand = samples[index][t]
            if t == 0:   
                m_sub[t][j].setObjective(vari_cost*q_sub[t][j] + unit_hold_cost*I_sub[t][j] + unit_back_cost*B_sub[t][j] +theta_sub[t][j], GRB.MINIMIZE)
                m_sub[t][j].addConstr(I_sub[t][j] - B_sub[t][j] == ini_I + q_value - demand)
                print('')               
            else:
                if t == T - 1:                   
                    m_sub[t][j].setObjective(unit_hold_cost*I_sub[t][j] + unit_back_cost*B_sub[t][j], GRB.MINIMIZE)
                else:
                    m_sub[t][j].setObjective(vari_cost*q_sub[t][j] + unit_hold_cost*I_sub[t][j] + unit_back_cost*B_sub[t][j] +theta_sub[t][j], GRB.MINIMIZE)
                last_index = 0
                for k in node_index[t - 1]:
                    if node_index[t][j][0] in k:
                        last_index = node_index[t - 1].index(k)
                m_sub[t][j].addConstr(I_sub[t][j] - B_sub[t][j] == I_sub_values[t-1][last_index] - B_sub_values[t-1][last_index] + q_detail_values[iter-1][t][last_index] - demand)
                print(end = '')
                    
            # optimize
            m_sub[t][j].optimize()
#           m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.lp')
#           m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.dlp')          
#           m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.sol')
            obj[j] = m_sub[t][j].objVal
            if t < T - 1:              
                q_detail_values[iter - 1][t+1][j] = q_sub[t][j].x
                
            I_sub_values[t][j] = I_sub[t][j].x 
            B_sub_values[t][j] = B_sub[t][j].x
            pi = m_sub[t][j].getAttr(GRB.Attr.Pi)
            pi_sub_detail_values[iter-1][t][j] = pi
            rhs = m_sub[t][j].getAttr(GRB.Attr.RHS)
            if t < T - 1:
                num_con = len(pi)
                for k in range(num_con-1):
                    pi_rhs_values[t][j] += pi[k]*rhs[k]
                pi_rhs_values[t][j] += -pi[-1]*demand 
            else:
                pi_rhs_values[t][j] = -pi[-1] * demand
            pi_values[t][j] = pi[-1]
            d_sub_values[t][j] = demand
            m_sub[t][j].remove(m_sub[t][j].getConstrs()[-1])
            for k in node_index[t][j]:
                pi_values2[t][k] = pi[-1]
                pi_rhs_values2[t][k] = pi_rhs_values[t][j]
            
            
            # # get slope and intercept
            # slope[j] = pi[0]
            # if t == 0:
            #     intercept[j] = obj[j] - pi[0] * q_value
            # else:
            #     intercept[j] = obj[j] - pi[0] * (I_sub_values[t-1][last_index]- B_sub_values[t-1][last_index] + q_detail_values[iter-1][t][last_index])
            
            
        # get and add the cut  
        # cut method 1     
        # actually every node in stage t share the same cut
        # this is not the formal handling of NLDS, nor the formal handling of SDDP
        # but the result seems close to optimal       
        
        avg_pi = sum(pi_values[t]) / t_nodeNum[t]
        sum_pi_rhs = 0
        for j in range(t_nodeNum[t]): 
            sum_pi_rhs += pi_rhs_values[t][j]
        avg_pi_rhs = sum_pi_rhs / t_nodeNum[t]
        if t == 0:
            m.addConstr(theta >= avg_pi*q + avg_pi_rhs) # just the benders optimality cut, same as the below constraint
            # m.write('test.lp')
           
        else:
            for j in range(t_nodeNum[t-1]):                  
                m_sub[t-1][j].addConstr(theta_sub[t-1][j] >= avg_pi*(I_sub[t-1][j] - B_sub[t-1][j] + q_sub[t-1][j]) + avg_pi_rhs)
                m_sub[t-1][j].update()
                # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.lp')
                print(end='')
        
        # cut method 2
        # formal handling of NLDS
        
        if t == 0:
            avg_pi = sum(pi_values[t]) / t_nodeNum[t]
            sum_pi_rhs = 0
            for j in range(t_nodeNum[t]): 
                sum_pi_rhs += pi_rhs_values[t][j]
            avg_pi_rhs = sum_pi_rhs / t_nodeNum[t]
            m.addConstr(theta >= avg_pi*q + avg_pi_rhs) # just the benders optimality cut, same as the below constraint
            # m.write('test.lp')
        else:
            for j in range(t_nodeNum[t-1]):  
                sum_pi = 0
                sum_pi_rhs = 0
                for k in node_index[t-1][j]:
                    sum_pi += pi_values2[t][k]
                    sum_pi_rhs += pi_rhs_values2[t][k]
                avg_pi = sum_pi / len(node_index[t-1][j])
                avg_pi_rhs = sum_pi_rhs / len(node_index[t-1][j])
                m_sub[t-1][j].addConstr(theta_sub[t-1][j] >= avg_pi*(I_sub[t-1][j] - B_sub[t-1][j] + q_sub[t-1][j]) + avg_pi_rhs)
               
                    
        
        # cut method 3
        # avg_slope = sum(slope) / t_nodeNum[t]
        # avg_intercept = sum(intercept) / t_nodeNum[t]
        # if t == 0:
        #     m.addConstr(theta >= avg_slope * q + avg_intercept)
        #     m.write('test2.lp')
        # else:
        #     for j in range(t_nodeNum[t-1]): 
        #         m_sub[t-1][j].addConstr(theta_sub[t-1][j] >= avg_slope * (I_sub[t-1][j] - B_sub[t-1][j] + q_sub[t-1][j]) + avg_intercept)
        #         m_sub[t-1][j].update()
        #         # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.lp')
        # print(end='')
    iter += 1

end = time.process_time()
print('********************************************')
print('final expected total costs is %.2f' % z)
print('ordering Q in the first peiod is %.2f' % q_value)
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)



