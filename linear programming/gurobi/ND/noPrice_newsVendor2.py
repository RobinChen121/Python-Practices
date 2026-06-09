#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:31:46 2023

@author: zhenchen

@disp:  use sddp to solve traditional newsvendor problem without price.

sampling is in the while loop--slow
    
    
"""

import numpy as np
import scipy.stats as st
from gurobipy import *
import time
from functools import reduce
import itertools
import random
import time
 
    
def generate_sample(sample_num, trunQuantile, mu):
    samples = [0 for i in range(sample_num)]
    for i in range(sample_num):
        rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
        samples[i] = st.poisson.ppf(rand_p, mu)
    return samples

# get the number of elements in a list of lists
def getSizeOfNestedList(listOfElem):
    ''' Get number of elements in a nested list'''
    count = 0
    # Iterate over the list
    for elem in listOfElem:
        # Check if type of element is list
        if type(elem) == list:  
            # Again call this function to get the size of this element
            count += getSizeOfNestedList(elem)
        else:
            count += 1    
    return count

def get_tree_strcture(samples):
    T = len(samples[0])
    N = len(samples)
    node_values = [[] for t in range(T)]
    node_index = [[] for t in range(T)] # this is the wanted value
    for t in range(T):
        node_num = 0
        if t == 0:           
            for i in range(N):           
                if samples[i][t] not in node_values[t]:
                    node_values[t].append(samples[i][t]) 
                    node_index[t].append([])
                    node_index[t][node_num].append(i)
                    node_num = node_num + 1
                else:
                    temp_m = len(node_values[t])
                    for j in range(temp_m): # should revise
                        if samples[i][t] == node_values[t][j]:
                            node_index[t][j].append(i)
                            break
        else:
            lastNodeNum = len(node_index[t-1])
            for i in range(lastNodeNum):
                child_num = len(node_index[t-1][i])
                node_values[t].append([])
                for j in range(child_num):
                    index = node_index[t-1][i][j]
                    if samples[index][t] not in node_values[t][i]:
                        node_values[t][i].append(samples[index][t]) 
                        node_index[t].append([])
                        node_index[t][node_num].append(index)
                        node_num = node_num + 1
                    else:
                        temp_m = len(node_values[t][i]) #2
                        for k in range(temp_m): 
                            if samples[index][t] == node_values[t][i][k]:
                                node_index[t][k].append(index)
                                break
                    
    return node_values, node_index


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
# sample_nums = [2, 2]
scenarios = list(itertools.product(*samples_detail)) 
# sampling in the loop


theta_iniValue = 0 # initial theta values in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = theta_iniValue*T, vtype = GRB.CONTINUOUS, name = 'theta_2')


iter = 1
iter_num = 8 

# models and decision variables from stage 2 to stage T+1 in the backward process
m2_sub = [[Model() for j in range(sample_nums[t])] for t in range(T)] 
q2_sub = [[m2_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(j+1)) for j in range(sample_nums[t])] for t in range(T-1)]
I2_sub = [[m2_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(j+1)) for j in range(sample_nums[t])] for t in range(T)]
B2_sub = [[m2_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(j+1)) for j in range(sample_nums[t])] for t in range(T)]
theta2_sub = [[m2_sub[t][j].addVar(lb = theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(j+1)) for j in range(sample_nums[t])] for t in range(T-1)]

slope = [[ 0 for t in range(T)] for i in range(iter_num)]
intercept = [[ 0 for t in range(T)] for i in range(iter_num)]
while iter <= iter_num:   
    
    # sampling
    sample_num = 30
    # samples = [(5, 5), (5, 15), (15, 5), (15, 15)]
    samples= random.sample(scenarios, sample_num) # sampling without replacement
    samples.sort() # sort to make same numbers together
    node_values, node_index = get_tree_strcture(samples)
    
    # number of sampled nodes in each period
    t_nodeNum = [0 for i in range(T)]
    for t in range(T):
        t_nodeNum[t] = getSizeOfNestedList(node_values[t])
        
    # linear programming models in the forward process
    # decision variables from stage 2 to stage T+1
    m_sub = [[Model() for j in range(t_nodeNum[t])] for t in range(T)] 
    q_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]
    I_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
    B_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T)]
    theta_sub = [[m_sub[t][j].addVar(lb = theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(j+1)) for j in range(t_nodeNum[t])] for t in range(T-1)]
    
    q_detail_values = [[] for t in range(T)]
    for t in range(T):
        if t == 0:
            q_detail_values[t] = 0
        else:
            q_detail_values[t] = [0 for s in range(t_nodeNum[t-1])]
                
    # forward computation    
    # solve the first stage model    
    m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)
    m.update()
    m.optimize()
    m.write('iter' + str(iter) + '_main.lp')    
    m.write('iter' + str(iter) + '_main.sol')

    print(end = '')
    q_value = q.x
    q_detail_values[0] = q_value
    theta_value = theta.x
    z = m.objVal
    
    I_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    B_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)] 
    d_sub_values = [[0 for s in range(t_nodeNum[t])] for t in range(T)]
    
    pi_sub_values = [[0 for s in range(sample_nums[t])] for t in range(T)] 
    pi_rhs_values = [[0 for s in range(sample_nums[t])] for t in range(T)] 
    
    # forward
    for t in range(T):       
        for j in range(t_nodeNum[t]): 
            obj = [0.0 for i in range(t_nodeNum[t])] 
            index = node_index[t][j][0] # 0 is enough to ascertain the node value
            demand = samples[index][t] # demand in the node
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
                m_sub[t][j].addConstr(I_sub[t][j] - B_sub[t][j] == I_sub_values[t-1][last_index] - B_sub_values[t-1][last_index] + q_detail_values[t][last_index] - demand)
            if iter == 2:
                pass
            if iter > 1 and t < T - 1:
                for k in range(iter-1):  
                    m_sub[t][j].addConstr(theta_sub[t][j] >= slope[k][t]*(I_sub[t][j] - B_sub[t][j] + q_sub[t][j]) + intercept[k][t])
                    m_sub[t][j].update()
                        
            print(end = '')
                    
            # optimize
            m_sub[t][j].optimize()
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.lp')
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.dlp')           
            # m_sub[t][j].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.sol')
            obj[j] = m_sub[t][j].objVal
            if t < T - 1:              
                q_detail_values[t+1][j] = q_sub[t][j].x
                
            I_sub_values[t][j] = I_sub[t][j].x 
            B_sub_values[t][j] = B_sub[t][j].x
            d_sub_values[t][j] = demand
            
    # backward and add the cut  
    if iter==2:
        pass
    for t in range(T-1, -1, -1):
        for j in range(sample_nums[t]): 
            demand = samples_detail[t][j]
            if t == 0:   
                m2_sub[t][j].setObjective(vari_cost*q2_sub[t][j] + unit_hold_cost*I2_sub[t][j] + unit_back_cost*B2_sub[t][j] +theta2_sub[t][j], GRB.MINIMIZE)                               
            else:
                if t == T - 1:                   
                    m2_sub[t][j].setObjective(unit_hold_cost*I2_sub[t][j] + unit_back_cost*B2_sub[t][j], GRB.MINIMIZE)
                else:
                    m2_sub[t][j].setObjective(vari_cost*q2_sub[t][j] + unit_hold_cost*I2_sub[t][j] + unit_back_cost*B2_sub[t][j] +theta2_sub[t][j], GRB.MINIMIZE)
                
                # whether those nodes are same in cuts? 
                last_index = range(t_nodeNum[t-1])[0] # parent node index
                                                      # random select the 1st one
            if iter > 1 and t < T - 1:
                for k in range(iter - 1):            
                    m2_sub[t][j].addConstr(theta2_sub[t][j] >= slope[k][t]*(I2_sub[t][j] - B2_sub[t][j] + q2_sub[t][j]) + intercept[k][t])
                    m2_sub[t][j].update()
                    print(end='') 
            if t == 0:
                m2_sub[t][j].addConstr(I2_sub[t][j] - B2_sub[t][j] == ini_I + q_value - demand)
            else:
                m2_sub[t][j].addConstr(I2_sub[t][j] - B2_sub[t][j] == I_sub_values[t-1][last_index] - B_sub_values[t-1][last_index] + q_detail_values[t][last_index] - demand)
            
            
            # optimize and get the dual
            m2_sub[t][j].optimize()
            m2_sub[t][j].write('back-iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.lp')
            m2_sub[t][j].write('back-iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.dlp')          
            m2_sub[t][j].write('back-iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(j+1) + '.sol')
            pi = m2_sub[t][j].getAttr(GRB.Attr.Pi)
            rhs = m2_sub[t][j].getAttr(GRB.Attr.RHS)
            if t < T - 1:
                num_con = len(pi)
                for k in range(num_con-1):
                    pi_rhs_values[t][j] += pi[k]*rhs[k] # for previous cuts                    
                pi_rhs_values[t][j] += -pi[-1]*demand 
            else:
                pi_rhs_values[t][j] = -pi[-1] * demand
            pi_sub_values[t][j] = pi[-1]
            m2_sub[t][j].remove(m2_sub[t][j].getConstrs()[-1])

        # get and add the cut            
        avg_pi = sum(pi_sub_values[t]) / sample_nums[t]
        sum_pi_rhs = 0
        for j in range(sample_nums[t]): 
            sum_pi_rhs += pi_rhs_values[t][j]
        avg_pi_rhs = sum_pi_rhs / sample_nums[t]
        slope[iter-1][t] = avg_pi
        intercept[iter-1][t] = avg_pi_rhs
        
        if t == 0:
            m.addConstr(theta >= avg_pi*q + avg_pi_rhs) # just the benders optimality cut, same as the above constraint
            print(end='')
        else:
            for j in range(sample_nums[t-1]):             
                m2_sub[t-1][j].addConstr(theta2_sub[t-1][j] >= avg_pi*(I2_sub[t-1][j] - B2_sub[t-1][j] + q2_sub[t-1][j]) + avg_pi_rhs)
                m2_sub[t-1][j].update()
                print(end='')
    
    iter += 1

end = time.process_time()
print('********************************************')
print('final expected total costs is %.2f' % z)
print('ordering Q in the first peiod is %.2f' % q_value)
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time)
    
    
    