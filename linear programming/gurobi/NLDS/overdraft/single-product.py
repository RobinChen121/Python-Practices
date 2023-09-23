#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:52:57 2023

@author: zhenchen

@disp:  single product cash flow with overdraft, no lead time
    
    
"""


import sys 
sys.path.append("../..") 
from tree import *
from gurobipy import *
import time
from functools import reduce
import itertools
import random
import time
import numpy as np


start = time.process_time()


ini_I = 0
ini_cash = 0
vari_cost = 1
unit_sal = 0.5
price = 5
mean_demands = [10, 10]
overhead_cost = [50, 50]
r0 = 0.01
r1 = 0.1
T = len(mean_demands)

sample_nums = [10 for t in range(T)]
trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = reduce(lambda x, y: x * y, sample_nums, 1)

# samples_detail is the detailed samples in each period
samples_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    samples_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])

samples_detail = [[5, 15], [5, 15]]
scenarios = list(itertools.product(*samples_detail)) 
sample_num = 4

# sampling can't be in the while looping
samples= random.sample(scenarios, sample_num) # sampling without replacement
samples.sort() # sort to make same numbers together
node_values, node_index = get_tree_strcture(samples)

theta_iniValue = -400 # initial theta values in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
W0 = m.addVar(vtype = GRB.CONTINUOUS, name = 'W_0')
W1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'W_1')
theta = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_2')

# number of nodes in each period
t_nodeNum = [0 for i in range(T)]
for t in range(T):
    t_nodeNum[t] = getSizeOfNestedList(node_values[t])
    
iter = 0
iter_num = 6
q_detail_values = [[[] for t in range(T)] for iter in range(iter_num)] 
W0_detail_values = [[[] for t in range(T)] for iter in range(iter_num)] 
W1_detail_values = [[[] for t in range(T)] for iter in range(iter_num)] 
for i in range(iter_num):
    for t in range(T):
        if t == 0:
            q_detail_values[i][t] = 0
            W0_detail_values[i][t] = 0
            W1_detail_values[i][t] = 0
        else:
            q_detail_values[i][t] = [0 for s in range(t_nodeNum[t-1])]
            W0_detail_values[i][t] = [0 for s in range(t_nodeNum[t-1])]
            W1_detail_values[i][t] = [0 for s in range(t_nodeNum[t-1])]

slopes1 = [[[0 for n in range(t_nodeNum[t])] for t in range(T)] for iter in range(iter_num)] 
slopes2 = [[[0 for n in range(t_nodeNum[t])] for t in range(T)] for iter in range(iter_num)] 
intercept = [[[0 for n in range(t_nodeNum[t])] for t in range(T)] for iter in range(iter_num)] 
while iter < iter_num: 
    
    # sub computation    
    # solve the first stage model    
    m.setObjective(overhead_cost[0] + vari_cost*q + r1* W1 - r0* W0 + theta, GRB.MINIMIZE)
    m.addConstr(theta >= theta_iniValue*(T))
    m.addConstr(-vari_cost*q - W0 + W1 == overhead_cost[0] - ini_cash)
    
    m.optimize()
    # m.write('iter' + str(iter) + '_main.lp')
    # m.write('iter' + str(iter) + '_main.sol')
    
    print(end = '')
    q_value = q.x
    q_detail_values[iter][0] = q_value
    W0_value = W0.x
    W1_value = W1.x
    theta_value = theta.x
    z = m.objVal
    
    # decision variables from stage 2 to stage T+1
    m_sub = [[Model() for j in range(t_nodeNum[t])] for t in range(T)] 
    q_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]
    I_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
    B_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
    C_sub = [[m_sub[t][j].addVar(lb = -GRB.INFINITY, name = 'C_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
    # the number of W0, W1 is similar to q
    W0_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]
    W1_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]
    theta_sub = [[m_sub[t][j].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]

    
    I_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    B_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    C_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    d_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)]
    
    pi_values1 = [[0 for s in range(sample_num)] for t in range(T)]
    pi_values2 = [[0 for s in range(sample_num)] for t in range(T)]
    pi_rhs_values = [[0 for s in range(sample_num)] for t in range(T)] 
    
    # sub and backward  
    for t in range(T):       
        for n in range(t_nodeNum[t]): 
            index = node_index[t][n][0]
            demand = samples[index][t]
            
            if t == T - 1:                   
                m_sub[t][n].setObjective(-price*(demand - B_sub[t][n]) - unit_sal*I_sub[t][n], GRB.MINIMIZE)
            else:
                m_sub[t][n].setObjective(-price*(demand - B_sub[t][n]) + overhead_cost[t+1] + vari_cost*q_sub[t][n] + r1*W1_sub[t][n] - r0*W0_sub[t][n] + theta_sub[t][n], GRB.MINIMIZE) # 
            
            # constraints
            for k in node_index[t - 1]:
                if node_index[t][n][0] in k:
                    last_index = node_index[t - 1].index(k)
            if t < T - 1:
                m_sub[t][n].addConstr(theta_sub[t][n] >= theta_iniValue*(T))
                m_sub[t][n].addConstr(C_sub[t][n] - vari_cost*q_sub[t][n] - W0_sub[t][n] + W1_sub[t][n] == overhead_cost[t])
                     
            if t == 0:
                m_sub[t][n].addConstr(I_sub[t][n] - B_sub[t][n] == ini_I + q_value - demand)
            else:
                m_sub[t][n].addConstr(I_sub[t][n] - B_sub[t][n] == I_sub_values[t-1][last_index] + q_detail_values[iter][t][last_index] - demand)
            if t == 0:
                m_sub[t][n].addConstr(C_sub[t][n] + price*B_sub[t][n] == ini_cash - overhead_cost[t] - vari_cost*q_value\
                                          -r1*W1_detail_values[iter][0] + r0*W0_detail_values[iter][0] + price*demand)
            else:
                m_sub[t][n].addConstr(C_sub[t][n] + price*B_sub[t][n] == C_sub_values[t-1][last_index]- overhead_cost[t] - vari_cost*q_detail_values[iter][t][last_index]\
                                          -r1*W1_detail_values[iter][t][last_index]\
                                          + r0*W0_detail_values[iter][t][last_index] + price*demand) 
           
                   
            
            # optimize
            m_sub[t][n].optimize()
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.lp')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.dlp')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.sol')
            
            if t < T - 1:              
                q_detail_values[iter][t+1][n] = q_sub[t][n].x
                W0_detail_values[iter][t+1][n] = W0_sub[t][n].x
                W1_detail_values[iter][t+1][n] = W1_sub[t][n].x
                
            I_sub_values[t][n] = I_sub[t][n].x 
            B_sub_values[t][n] = B_sub[t][n].x
            C_sub_values[t][n] = C_sub[t][n].x

            pi = m_sub[t][n].getAttr(GRB.Attr.Pi)
            rhs = m_sub[t][n].getAttr(GRB.Attr.RHS)
            
            con_num = len(pi)
            pi_rhs = 0
            for k in range(con_num - 2): 
                pi_rhs += pi[k]*rhs[k]             
            pi_rhs += -pi[-2] * demand - pi[-1]*overhead_cost[t] + pi[-1]*price*demand - price*demand  # put here is better because of demand
            if t < T - 1:
                pi_rhs += overhead_cost[t]
            if t == 0:
                pi_rhs += pi[-1] * ini_cash
                pi_rhs += pi[-2] * ini_I
                
            for k in node_index[t][n]:
                pi_values1[t][k] = pi[-2]
                pi_values2[t][k] = pi[-1]
                pi_rhs_values[t][k] = pi_rhs
            
        if t == 0:
            avg_pi1 = sum(pi_values1[t]) / sample_num
            avg_pi2 = sum(pi_values2[t]) / sample_num
            avg_pi_rhs = sum(pi_rhs_values[t]) / sample_num
            m.addConstr(theta >= avg_pi1*q + avg_pi2*(-vari_cost*q - r1* W1 + r0* W0) + avg_pi_rhs)
        else:
            for n in range(t_nodeNum[t-1]): 
                sum_pi1 = 0
                sum_pi2 = 0
                sum_pi_rhs = 0
                for k in node_index[t-1][n]:
                    sum_pi1 += pi_values1[t][k]
                    sum_pi2 += pi_values2[t][k]
                    sum_pi_rhs += pi_rhs_values[t][k]
                avg_pi1 = sum_pi1 / len(node_index[t-1][n])
                avg_pi2 = sum_pi1 / len(node_index[t-1][n])
                avg_pi_rhs = sum_pi_rhs / len(node_index[t-1][n])
                m_sub[t-1][n].addConstr(theta_sub[t-1][n] >= avg_pi1*(I_sub[t-1][n] + q_sub[t-1][n])+ avg_pi2*(-vari_cost*q_sub[t-1][n] - r1* W1_sub[t-1][n] + r0* W0_sub[t-1][n]) + avg_pi_rhs)
                  
    iter += 1


