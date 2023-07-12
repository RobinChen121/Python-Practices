#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:52:57 2023

@author: zhenchen

@disp:  single product cash flow with overdraft, no lead time
    
    
"""


from sample_tree import generate_sample, get_tree_strcture, getSizeOfNestedList
from gurobipy import *
import time
from functools import reduce
import itertools
import random
import time


start = time.process_time()
ini_I = 0
ini_cash = 30
vari_cost = 1
price = 5
unit_back_cost = 0
mean_demands = [5, 5, 15]
T = len(mean_demands)
overhead_cost = [25 for t in range(T)]
draft_limit = 100
loan_rate = 0.03
deposite_rate = 0.003

sample_nums = [10 for t in range(T)]
trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = reduce(lambda x, y: x * y, sample_nums, 1)

# samples_detail is the detailed samples in each period
samples_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    samples_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])

# samples_detail = [[5, 15], [5, 15]]
scenarios = list(itertools.product(*samples_detail)) 
sample_num = 30

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

iter = 1
iter_num = 6
pi_sub_detail_values = [[[[] for s in range(t_nodeNum[t])] for t in range(T)] for iter in range(iter_num)] 
rhs_sub_detail_values = [[[[] for s in range(t_nodeNum[t])] for t in range(T)] for iter in range(iter_num)] 
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

result_iter = []
while iter <= iter_num: 
    
    # forward computation    
    # solve the first stage model    
    m.setObjective(overhead_cost[0] + vari_cost*q + loan_rate * W1 - deposite_rate * W0 + theta, GRB.MINIMIZE)
    m.addConstr(theta >= theta_iniValue*(T))
    m.addConstr(-vari_cost*q - loan_rate*W1 + deposite_rate*W0 >= overhead_cost[0] - ini_cash - draft_limit)
    m.addConstr(-vari_cost*q - W0 + W1 == overhead_cost[0] - ini_cash)
    
    m.optimize()
    m.write('iter' + str(iter) + '_main.lp')
    m.write('iter' + str(iter) + '_main.sol')
    
    print(end = '')
    q_value = q.x
    W0_value = W0.x
    W1_value = W1.x
    q_detail_values[iter - 1][0] = q_value
    theta_value = theta.x
    z = m.objVal
    
    I_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    B_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    C_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    pi_rhs_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    d_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)]
    
    # forward and backward  
    for t in range(T):       
        for j in range(t_nodeNum[t]): 
            index = node_index[t][j][0]
            demand = samples[index][t]
            if t == 0:   
                if t != T - 1: # wheter should add theta
                    m_sub[t][j].setObjective(vari_cost*q_sub[t][j] - price*(demand - B_sub[t][j]) +theta_sub[t][j], GRB.MINIMIZE)
                    m_sub[t][j].addConstr(theta_sub[t][j] >= theta_iniValue*(T-1-t))
                else:
                    m_sub[t][j].setObjective(- price*(demand - B_sub[t][j]), GRB.MINIMIZE)
                if t < T - 1:
                    m_sub[t][j].addConstr(vari_cost*q_sub[t][j] <= C_sub[t][j])      
                m_sub[t][j].addConstr(I_sub[t][j] - B_sub[t][j] == ini_I + q_value - demand)
                m_sub[t][j].addConstr(C_sub[t][j] == ini_cash + price*(demand - B_sub[t][j]) - vari_cost*q_value)  
            else:
                if t == T - 1:                   
                    m_sub[t][j].setObjective(- price*(demand - B_sub[t][j]), GRB.MINIMIZE)
                else:
                    m_sub[t][j].setObjective(vari_cost*q_sub[t][j] - price*(demand - B_sub[t][j]) + theta_sub[t][j], GRB.MINIMIZE)
                    m_sub[t][j].addConstr(theta_sub[t][j] >= theta_iniValue*(T-1-t))
                last_index = 0
                for k in node_index[t - 1]:
                    if node_index[t][j][0] in k:
                        last_index = node_index[t - 1].index(k)
                if t < T - 1:
                    m_sub[t][j].addConstr(vari_cost*q_sub[t][j] <= C_sub[t][j])
                pass
                m_sub[t][j].addConstr(I_sub[t][j] - B_sub[t][j] == I_sub_values[t-1][last_index] + q_detail_values[iter-1][t][last_index] - demand)
                m_sub[t][j].addConstr(C_sub[t][j] == C_sub_values[t-1][last_index] + price*(demand - B_sub[t][j]) - vari_cost*q_detail_values[iter-1][t][last_index])
                
            
            # optimize
            m_sub[t][j].optimize()
            if t < T - 1 and theta_sub[t][j].x != theta_iniValue*(T-1-t):
                print()
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.lp')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.dlp')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.sol')
    
    
    iter += 1


