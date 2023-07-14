#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 23:31:56 2023

@author: zhenchen

@disp:  sddp for multi period newsvendor, lost sale variable B in the objective function;
 
more length of the planning horizon, more iterations to converge;
have backorder;
    
    
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
vari_cost = 1
price = 10
unit_back_cost = 10
unit_hold_cost = 2
mean_demands = [10, 20]
sample_nums = [10, 10]
T = len(mean_demands)
trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = reduce(lambda x, y: x * y, sample_nums, 1)

# samples_detail is the detailed samples in each period
samples_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    samples_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])

# samples_detail = [[5, 15], [5, 15], [5, 15]]
scenarios = list(itertools.product(*samples_detail)) 
sample_num = 30
samples= random.sample(scenarios, sample_num) # sampling without replacement
samples.sort() # sort to make same numbers together
node_values, node_index = get_tree_strcture(samples)

theta_iniValue = -300 # initial theta values in each period
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
theta_sub = [[m_sub[t][j].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]

iter = 1
iter_num = 15
pi_sub_detail_values = [[[[] for s in range(t_nodeNum[t])] for t in range(T)] for iter in range(iter_num)] 
rhs_sub_detail_values = [[[[] for s in range(t_nodeNum[t])] for t in range(T)] for iter in range(iter_num)] 
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
    m.write('iter' + str(iter) + '_main.lp')
    m.write('iter' + str(iter) + '_main.sol')
    
    print(end = '')
    q_value = q.x
    q_detail_values[iter - 1][0] = q_value
    theta_value = theta.x
    z = m.objVal
    
    I_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    B_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    pi_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    pi_rhs_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    d_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)]
    pi_rhs_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    # forward and backward  
    for t in range(T):       
        for j in range(t_nodeNum[t]): 
            obj = [0.0 for i in range(t_nodeNum[t])] 
            index = node_index[t][j][0]
            demand = samples[index][t]
            if t == 0:   
                if T > 1:
                    m_sub[t][j].setObjective(vari_cost*q_sub[t][j] + unit_hold_cost*I_sub[t][j] - price*(demand - B_sub[t][j]) +theta_sub[t][j], GRB.MINIMIZE)
                    m_sub[t][j].addConstr(theta_sub[t][j] >= theta_iniValue*(T-1-t))
                else:
                    m_sub[t][j].setObjective(unit_hold_cost*I_sub[t][j] - price*(demand - B_sub[t][j]), GRB.MINIMIZE)
                m_sub[t][j].addConstr(I_sub[t][j] - B_sub[t][j] == ini_I + q_value - demand)
                print('')               
            else:
                if t == T - 1:                   
                    m_sub[t][j].setObjective(unit_hold_cost*I_sub[t][j] - price*(demand - B_sub[t][j]), GRB.MINIMIZE)
                else:
                    m_sub[t][j].setObjective(vari_cost*q_sub[t][j] + unit_hold_cost*I_sub[t][j] - price*(demand - B_sub[t][j]) +theta_sub[t][j], GRB.MINIMIZE)
                    m_sub[t][j].addConstr(theta_sub[t][j] >= theta_iniValue*(T-1-t))
                last_index = 0
                for k in node_index[t - 1]:
                    if node_index[t][j][0] in k:
                        last_index = node_index[t - 1].index(k)
                m_sub[t][j].addConstr(I_sub[t][j] - B_sub[t][j] == I_sub_values[t-1][last_index] - B_sub_values[t-1][last_index] + q_detail_values[iter-1][t][last_index] - demand)
                print(end = '')
                    
            # optimize
            m_sub[t][j].optimize()
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.lp')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.dlp')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.sol')
            obj[j] = m_sub[t][j].objVal
            if t < T - 1:              
                q_detail_values[iter - 1][t+1][j] = q_sub[t][j].x
                
            I_sub_values[t][j] = I_sub[t][j].x 
            B_sub_values[t][j] = B_sub[t][j].x
            pi = m_sub[t][j].getAttr(GRB.Attr.Pi)
            pi_sub_detail_values[iter-1][t][j] = pi
            rhs = m_sub[t][j].getAttr(GRB.Attr.RHS)
            rhs_sub_detail_values[iter-1][t][j] = rhs
            
            if iter == 2:
                pass
            # if t < T - 1:
            num_con = len(pi)
            for k in range(num_con - 1): # all the previous constraints
                pi_rhs_values[t][j] += pi[k]*rhs[k] # should not include the inventory flow constrints (q inside)
            pi_rhs_values[t][j] += -pi[-1] * demand - price*demand # the inventory flow constraints
        # else:
            #     pi_rhs_values[t][j] = -pi[-1] * demand - price*demand
            pi_sub_values[t][j] = pi[-1]
            d_sub_values[t][j] = demand
            # so hyperplane cuts are always in the front
            m_sub[t][j].remove(m_sub[t][j].getConstrs()[-1]) # inventory flow
            if t < T - 1:
                m_sub[t][j].remove(m_sub[t][j].getConstrs()[-2]) # theta bound constraint
            
        # get and add the cut      
        # very important
        if iter == 2:
            pass
        avg_pi = sum(pi_sub_values[t]) / t_nodeNum[t]
        sum_pi_rhs = 0
        for j in range(t_nodeNum[t]): 
            sum_pi_rhs += pi_rhs_values[t][j]
        avg_pi_rhs = sum_pi_rhs / t_nodeNum[t]
        if t == 0:
            # should have more
            m.addConstr(theta >= avg_pi*q + avg_pi_rhs) # just the benders optimality cut, same as the above constraint
            # m.write('test.lp')
            print(end='')
        else:
            for j in range(t_nodeNum[t-1]):             
                m_sub[t-1][j].addConstr(theta_sub[t-1][j] >= avg_pi*(I_sub[t-1][j] + q_sub[t-1][j]) + avg_pi_rhs)
                m_sub[t-1][j].update()
                print(end='')
    
    iter += 1

end = time.process_time()
print('********************************************')
print('final expected total costs is %.2f' % z)
print('ordering Q in the first peiod is %.2f' % q_value)
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)

