# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 21:10:47 2022

@author: chen
"""

import numpy as np
import scipy.stats as st
from gurobipy import Model, GRB
import time
from functools import reduce
import itertools
import random
 
    
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


ini_I = 0
ini_cash = 0
price  =  6
vari_cost = 2
sal_value = 1
uni_hold_cost = 0 # no holding cost
mean_demands = [10, 10]
sample_nums = [10, 10]
T = len(mean_demands)
trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = reduce(lambda x, y: x * y, sample_nums, 1)

# samples_detail is the detailed samples in each period
samples_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    samples_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])

#samples_detail = [[8, 12], [8, 12]]
scenarios = list(itertools.product(*samples_detail)) 
S = round(scenario_numTotal)
sample_num = 30
samples= random.sample(scenarios, sample_num) # sampling without replacement
samples.sort() # sort to make same numbers together
node_values, node_index = get_tree_strcture(samples)

theta_iniValue = -1000 # initial theta values in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS)
theta = m.addVar(lb = theta_iniValue*T, vtype = GRB.CONTINUOUS)

# number of nodes in each period
t_nodeNum = [0 for i in range(T)]
for t in range(T):
    t_nodeNum[t] = getSizeOfNestedList(node_values[t])
# decision variables from stage 2 to stage T+1
m_sub = [[Model() for j in range(t_nodeNum[t])] for t in range(T)] 
q_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS) for j in range(t_nodeNum[t])] for t in range(T-1)]
I_sub = [[m_sub[t][j].addVar(vtype = GRB.CONTINUOUS) for j in range(t_nodeNum[t])] for t in range(T)]
theta_sub = [[m_sub[t][j].addVar(lb = theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS) for j in range(t_nodeNum[t])] for t in range(T-1)]

iter = 1
iter_num = 50
while iter < iter_num:   
    I_sub_value = [[0 for t in range(T)] for s in range(sample_num)]
    q_sub_value = [[0 for t in range(T-1)] for s in range(sample_num)]
        
    # forward computation    
    # solve the first stage model    
    m.setObjective(vari_cost*q+theta, GRB.MINIMIZE)
    m.update()
    m.write('test.lp')
    m.optimize()
    
    print(end='')
    q_value = q.x
    theta_value = theta.x
    z = m.objVal
    
    # initial value of the decision variables except the last period
    # if iter == 1:
    #     q_value = 10
    #     q_sub_value = [[10 for t in range(T-1)] for s in range(sample_num)]
        # for t in range(T-1):
        #     for j in range(t_nodeNum[t]):
        #         index_detail = node_index[t][j]
        #         for s in index_detail:
        #             demand = samples[s][t]
        #             if t == 0:
        #                 I_sub_value[s][t] = max(0, q_value - demand)
        #             else:
        #                 I_sub_value[s][t] = max(0, I_sub_value[s][t-1] + q_sub_value[s][t] - demand)
    
    # forward and backward
    for t in range(T):
        obj = [0.0 for i in range(t_nodeNum[t])] 
        pi = [0.0 for i in range(t_nodeNum[t])]
        pi1 = [0.0 for i in range(t_nodeNum[t])]
        Dpi2 = [0.0 for i in range(t_nodeNum[t])]
        
        for j in range(t_nodeNum[t]): 
            # objective and constraints
            index = node_index[t][j][0]
            demand = samples[index][t]
            if t == 0:   
                m_sub[t][j].setObjective(vari_cost*q_sub[t][j]-price*(q_value-I_sub[t][j])+theta_sub[t][j], GRB.MINIMIZE)
                m_sub[t][j].addConstr(I_sub[t][j] >= q_value - demand)
                m_sub[t][j].addConstr(-I_sub[t][j] >= -q_value)
                m_sub[t][j].write('test.lp')
                print('')               
            else:
                if t == T - 1:                   
                    m_sub[t][j].setObjective(-price*(q_sub_value[index][t-1]-I_sub[t][j])-sal_value*I_sub[t][j], GRB.MINIMIZE)
                    m_sub[t][j].write('test.lp')
                    m_sub[t][j].addConstr(I_sub[t][j] >= I_sub_value[index][t-1] + q_sub_value[index][t-1] - demand)
                    m_sub[t][j].addConstr(-I_sub[t][j] >= -I_sub_value[index][t-1] - q_sub_value[index][t-1])
                    m_sub[t][j].write('test.lp') # to see the linear programming model in the final stage
                    print(end='')  
                # else:                   
                    # m_sub[t][j].setObjective(vari_cost*q_sub[t][j]-price*(q_sub_value[index][t-1]-I_sub[t][j])+theta_sub[t][j], GRB.MINIMIZE)
                    # m_sub[t][j].addConstr(I_sub[t][j] >= I_sub_value[index][t-1] + q_sub_value[index][t-1] - demand)
                    
                    
            # optimize
            m_sub[t][j].optimize()
            obj[j] = m_sub[t][j].objVal
            m_sub[t][j].remove(m_sub[t][j].getConstrs()[-2:])
            if t < T - 1:
                index = node_index[t][j][0]
                q_sub_value[index][t] = q_sub[t][j].x
                I_sub_value[index][t] = I_sub[t][j].x 
            pi = m_sub[t][j].getAttr(GRB.Attr.Pi)
            pi1[j] = pi[-2] - pi[-1] 
            Dpi2[j] = pi[-2] * demand
            
        # get and add the cut        
        avg_obj = sum(obj) / t_nodeNum[t]
        avg_pi1 = sum(pi1) / t_nodeNum[t]
        avg_Dpi2 = sum(Dpi2)/t_nodeNum[t]
        if t == 0:
            m.addConstr(theta >= avg_pi1*q-price*q-avg_Dpi2-40) # just the benders optimality cut, same as the above constraint
            m.write('test.lp')
            print(end='')
        else:
            for j in range(t_nodeNum[t-1]):
                index = node_index[t-1][j][0]
                m_sub[t-1][j].addConstr(theta_sub[t-1][j] >= avg_pi1*(I_sub[t-1][j] + q_sub[t-1][j])-price*q_sub[t-1][j] -avg_Dpi2)
                m_sub[t-1][j].update()
                m_sub[t-1][j].write('test.lp')
                print(end='')
    iter += 1
print('ordering quantity is %.2f' % q_value)
print('second stage expected profit is %.2f' % theta_value)
print('expected profit is %.2f' % z)



