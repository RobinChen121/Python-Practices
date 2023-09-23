#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:55:53 2023

@author: zhenchen

@disp:  test sddp for cash constrained multi period problem
    
    
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
ini_cash = 10
vari_cost = 1
price = 10
unit_back_cost = 0
mean_demands = [10, 10]
sample_nums = [10, 10]
T = len(mean_demands)
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
C_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
theta_sub = [[m_sub[t][j].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]

iter = 1
iter_num = 4
pi_sub_detail_values = [[[[] for s in range(t_nodeNum[t])] for t in range(T)] for iter in range(iter_num)] 
rhs_sub_detail_values = [[[[] for s in range(t_nodeNum[t])] for t in range(T)] for iter in range(iter_num)] 
q_detail_values = [[[] for t in range(T)] for iter in range(iter_num)] 
for i in range(iter_num):
    for t in range(T):
        if t == 0:
            q_detail_values[i][t] = 0
        else:
            q_detail_values[i][t] = [0 for s in range(t_nodeNum[t-1])]

result_iter = []
while iter <= iter_num:       
    
    # forward computation    
    # solve the first stage model    
    m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)
    m.addConstr(theta >= theta_iniValue*(T))
    m.addConstr(vari_cost*q <= ini_cash)
    m.optimize()
    m.write('iter' + str(iter) + '_main.lp')
    # m.write('iter' + str(iter) + '_main.sol')
    
    print(end = '')
    q_value = q.x
    q_detail_values[iter - 1][0] = q_value
    theta_value = theta.x
    z = m.objVal
    
    I_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    B_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    C_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    pi_rhs_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    d_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)]
    pi_sub1_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)]
    pi_sub2_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)]
    pi_sub3_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)]
    pi_rhs_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    
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
            m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.lp')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.dlp')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.sol')

            if t < T - 1:              
                q_detail_values[iter - 1][t+1][j] = q_sub[t][j].x
                
            I_sub_values[t][j] = I_sub[t][j].x 
            B_sub_values[t][j] = B_sub[t][j].x
            C_sub_values[t][j] = C_sub[t][j].x
            pi = m_sub[t][j].getAttr(GRB.Attr.Pi)
            
            pi_sub_detail_values[iter-1][t][j] = pi
            
            rhs = m_sub[t][j].getAttr(GRB.Attr.RHS)
            rhs_sub_detail_values[iter-1][t][j] = rhs
            
            con_num = len(pi)
            for k in range(con_num - 2): 
                pi_rhs_values[t][j] += pi[k]*rhs[k] 
            
            pi_rhs_values[t][j] += -pi[-2] * demand + pi[-1] * price * demand - price*demand # put here is better because of demand
            if iter > 1:
                pass
            d_sub_values[t][j] = demand
            pi_sub1_values[t][j] = pi[-2] # inventory flow constraint
            pi_sub2_values[t][j] = pi[-1] # cash constraint
            m_sub[t][j].remove(m_sub[t][j].getConstrs()[-1]) 
            m_sub[t][j].remove(m_sub[t][j].getConstrs()[-2])         
            if t < T - 1:
                m_sub[t][j].remove(m_sub[t][j].getConstrs()[-3])
                m_sub[t][j].remove(m_sub[t][j].getConstrs()[-4])
            
        # get and add the cut  
        # not formal NLSD cut
        avg_pi1 = sum(pi_sub1_values[t]) / t_nodeNum[t]
        avg_pi2 = sum(pi_sub2_values[t]) / t_nodeNum[t]
        if t == 1:
            pass
        sum_pi_rhs = 0
        for j in range(t_nodeNum[t]): 
            sum_pi_rhs += pi_rhs_values[t][j]
        avg_pi_rhs = sum_pi_rhs / t_nodeNum[t]
        if iter > 1 and t == 0:
            print()
        if t == 0:
            m.addConstr(theta >= avg_pi1*(ini_I + q) + avg_pi2*ini_cash - avg_pi2*vari_cost*q + avg_pi_rhs) 
        else:
            for j in range(t_nodeNum[t-1]):  
                m_sub[t-1][j].addConstr(theta_sub[t-1][j] >= avg_pi1*(I_sub[t-1][j] + q_sub[t-1][j]) + avg_pi2*C_sub[t-1][j] - avg_pi2*vari_cost*q_sub[t-1][j] + avg_pi_rhs)
                m_sub[t-1][j].update()

    m.remove(m.getConstrs()[-2])  
    m.remove(m.getConstrs()[-1])  
    m.update()  
    result_iter.append(-z)
    iter += 1

end = time.process_time()
print('********************************************')
final_cash = -z
print('final expected cash increment is %.2f' % final_cash)
print('ordering Q in the first peiod is %.2f' % q_value)
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)