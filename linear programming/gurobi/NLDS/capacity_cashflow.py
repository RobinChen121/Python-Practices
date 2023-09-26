#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:55:53 2023

@author: zhenchen

@disp:  test NLSD for cash constrained multi period problem,

may be there are some errors
    
    
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
ini_cash = 0
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

iter = 0
iter_num = 7
q_detail_values = [[[] for t in range(T)] for iter in range(iter_num)] 
for i in range(iter_num):
    for t in range(T):
        if t == 0:
            q_detail_values[i][t] = 0
        else:
            q_detail_values[i][t] = [0 for s in range(t_nodeNum[t-1])]


m.addConstr(theta >= theta_iniValue*(T))
# m.addConstr(vari_cost*q <= ini_cash)
m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)

slope1 = [[0 for t in range(T-1)] for iter in range(iter_num)]
slope2 = [[0 for t in range(T-1)] for iter in range(iter_num)]
intercept = [[0 for t in range(T-1)] for iter in range(iter_num)]
while iter < iter_num:       
    
    # forward computation    
    # solve the first stage model    
        
    m.optimize()
    m.write('iter' + str(iter) + '_main.lp')
    m.write('iter' + str(iter) + '_main.sol')
    
    print(end = '')
    q_value = q.x
    q_detail_values[iter][0] = q_value
    theta_value = theta.x
    z = m.objVal
    
    # decision variables from stage 2 to stage T+1
    m_sub = [[Model() for j in range(t_nodeNum[t])] for t in range(T)] 
    q_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]
    I_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
    B_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
    C_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
    theta_sub = [[m_sub[t][j].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]

    I_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    B_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    C_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    
    pi_values1 = [[0 for s in range(sample_num)] for t in range(T)]
    pi_values2 = [[0 for s in range(sample_num)] for t in range(T)]
    pi_rhs_values = [[0 for s in range(sample_num)] for t in range(T)] 
    
    # forward and backward  
    for t in range(T):       
        for j in range(t_nodeNum[t]): 
            index = node_index[t][j][0]
            demand = samples[index][t]
            
            # add cust
            if t < T - 1:
                for k in range(iter):
                    if t == 0:
                        m_sub[t][j].addConstr(theta_sub[t][j] >= slope1[k][t]*(I_sub[t][j] + q_sub[t][j])+ slope2[k][t]*(-vari_cost*q_sub[t][j]) + intercept[k][t])
                    else:
                        m_sub[t][j].addConstr(theta_sub[t][j] >= slope1[k][t]*(I_sub[t][j] + q_sub[t][j])+ slope2[k][t]*(C_sub[t][j] -vari_cost*q_sub[t][j]) + intercept[k][t])
            
            if t < T - 1:
                m_sub[t][j].setObjective(vari_cost*q_sub[t][j] - price*(demand - B_sub[t][j]) +theta_sub[t][j], GRB.MINIMIZE)
                m_sub[t][j].addConstr(theta_sub[t][j] >= theta_iniValue*(T-1-t))
            else:
                m_sub[t][j].setObjective(- price*(demand - B_sub[t][j]), GRB.MINIMIZE)
                
            if t == 0:   
                m_sub[t][j].addConstr(I_sub[t][j] - B_sub[t][j] == ini_I + q_value - demand)
                m_sub[t][j].addConstr(C_sub[t][j] == ini_cash + price*(demand - B_sub[t][j]) - vari_cost*q_value)  
            else:
                last_index = 0
                for k in node_index[t - 1]:
                    if node_index[t][j][0] in k:
                        last_index = node_index[t - 1].index(k)
                
                m_sub[t][j].addConstr(I_sub[t][j] - B_sub[t][j] == I_sub_values[t-1][last_index] + q_detail_values[iter][t][last_index] - demand)
                m_sub[t][j].addConstr(C_sub[t][j] == C_sub_values[t-1][last_index] + price*(demand - B_sub[t][j]) - vari_cost*q_detail_values[iter][t][last_index])               
                # if t < T - 1:
                #     m_sub[t][j].addConstr(vari_cost*q_sub[t][j] <= C_sub[t][j])
 
                
            
            # optimize
            m_sub[t][j].update()
            m_sub[t][j].optimize()
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.lp')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.sol')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.dlp')
            
            if iter == 1 and t == 0:
                pass
            
            if t < T - 1:              
                q_detail_values[iter][t+1][j] = q_sub[t][j].x
                
            I_sub_values[t][j] = I_sub[t][j].x 
            B_sub_values[t][j] = B_sub[t][j].x
            C_sub_values[t][j] = C_sub[t][j].x
            pi = m_sub[t][j].getAttr(GRB.Attr.Pi)
            
            rhs = m_sub[t][j].getAttr(GRB.Attr.RHS)
            pi_rhs = 0
            con_num = len(pi)
            for k in range(con_num - 2): 
                pi_rhs += pi[k]*rhs[k] 
            
            pi_rhs += -pi[-2] * demand + pi[-1] * price * demand - price*demand # put here is better because of demand

            for k in node_index[t][j]:
                pi_values1[t][k] = pi[-2]
                pi_values2[t][k] = pi[-1]
                pi_rhs_values[t][k] = pi_rhs
            
        # get and add the cut  
        # formal NLSD cut
        if t == 0:
            avg_pi1 = sum(pi_values1[t]) / sample_num
            avg_pi2 = sum(pi_values2[t]) / sample_num
            avg_pi_rhs = sum(pi_rhs_values[t]) / sample_num
            m.addConstr(theta >= avg_pi1*(ini_I + q) + avg_pi2*(ini_cash - vari_cost*q) + avg_pi_rhs)            
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
                avg_pi2 = sum_pi2 / len(node_index[t-1][n])
                avg_pi_rhs = sum_pi_rhs / len(node_index[t-1][n])
                slope1[iter][t-1] = avg_pi1
                slope2[iter][t-1] = avg_pi2
                intercept[iter][t-1] = avg_pi_rhs
                pass
 
    iter += 1

end = time.process_time()
print('********************************************')
final_cash = -z
print('final expected cash increment is %.2f' % final_cash)
print('ordering Q in the first peiod is %.2f' % q_value)
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)