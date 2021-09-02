# -*- coding: utf-8 -*-
"""
Created on Wed May 19 17:31:27 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: A chance-constraint programming approach for cash-flow stochastic inventory flow problems;
             sort the scenarios and use the method proposed by Luedtke (2010) to compute faster.
             
             This method is not suitable for the SAA problem where decision variables depend on scenarios.
    
"""


import numpy as np
import scipy.stats as st
from gurobipy import *
import time
from functools import reduce
import itertools


def generate_sample(sample_num, trunQuantile, mus):
    T = len(mus)
    samples = [[0 for i in range(sample_num[t])] for t in range(T)]
    for t in range(T):
        for i in range(sample_num[t]):
            rand_p = np.random.uniform(trunQuantile*i/sample_num[t], trunQuantile*(i+1)/sample_num[t])
            samples[t][i] = st.poisson.ppf(rand_p, mus[t])
    return samples
        

# make s as first index
def get_sample_detail(samples, scenario_permulations):
    S = len(scenario_permulations)
    T = len(samples)
    samples2 = [[0 for t in range(T)] for s in range(S)]
    for s in range(S):
        index = scenario_permulations[s]
        for t in range(T):
            samples2[s][t] = samples[t][index[t]]
    return samples2

ini_cash = 5
ini_I = 0
price  =  6
vari_cost = 2
sal_value = 1
mean_demands = [15]
sample_nums = [20]
T = len(mean_demands)
service_rate = 0.8
overhead_costs = [0 for t in range(T)]
trunQuantile = 0.9999

S = reduce(lambda x, y: x * y, sample_nums, 1)
K = np.int(np.floor(S * (1 - service_rate))) # maximum number of scenarios that have negative cash

samples = generate_sample(sample_nums, trunQuantile, mean_demands)
arr = []
for t in range(T):
    arr.append(range(sample_nums[t]))
scenario_permulations = list(itertools.product(*arr))
sample_detail = get_sample_detail(samples, scenario_permulations)

# sort the array
# sample_detail.sort(key = lambda x: sum(x)) 
# sample_detail = np.array(sample_detail)


# test the MIP model again
tic = time.time()
try:
    # Create a new model
    m = Model("saa-chance")
    
    # Create variables          
    Q = [[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for t in range(T)] # ordering quantity
    delta = [[m.addVar(vtype = GRB.BINARY) for s in range(S)] for t in range(T)] # auxiliary variable
    I = [[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for t in range(T)] # inventory variable
    alpha = [m.addVar(vtype = GRB.BINARY) for s in range(S)] # whether cash balance is negative in this scenario
    beta = [[m.addVar(vtype = GRB.CONTINUOUS) for s in range(K)] for t in range(T)] # binary variable in extened formulation
    cash = [[LinExpr() for s in range(S)] for t in range(T)]
    min_cash = [[LinExpr() for s in range(S)] for t in range(T)]
    total_expect_cash = LinExpr()
    M1 = 10000
    M2 = 10000
      
    # cash flow
    for t in range(T):
        for s in range(S):
            if t == 0 and T > 1: # end-of-period cash balance in the first period
                cash[t][s] = ini_cash + price * (ini_I + Q[t][s] - I[t][s]) - vari_cost * Q[t][s] - overhead_costs[t]
            elif t == 0 and T <= 1:
                cash[t][s] = ini_cash + price * (ini_I + Q[t][s] - I[t][s]) - vari_cost * Q[t][s] - overhead_costs[t] + sal_value * I[t][s]
            elif t == T - 1 and T > 1:
                cash[t][s] = cash[t-1][s] + price * (I[t-1][s] + Q[t][s] - I[t][s]) - vari_cost * Q[t][s] - overhead_costs[t] + sal_value * I[t][s]
            else:
                cash[t][s] = cash[t-1][s] + price * (I[t-1][s] + Q[t][s] - I[t][s]) - vari_cost * Q[t][s] - overhead_costs[t]
            if t == 0:
                min_cash[t][s] = ini_cash - vari_cost * Q[t][s] 
            else:
                min_cash[t][s] = cash[t-1][s] - vari_cost * Q[t][s]           
            
            
    expect_cash = sum(cash[T - 1]) / S
    m.update()    
    
    # add constraints
    # inventory flow
    for s in range(S):
        for t in range(T):
            demand = samples[t][scenario_permulations[s][t]]  # be careful
            if t == 0:
                m.addConstr(I[t][s] <= ini_I + Q[t][s] - demand + (1 - delta[t][s]) * M1)     
                m.addConstr(I[t][s] >= ini_I + Q[t][s] - demand - (1 - delta[t][s]) * M1)   
                m.addConstr(ini_I + Q[t][s] - demand <= delta[t][s]* M1 -0.1) 
            else:
                try:
                    m.addConstr(I[t][s] <= I[t-1][s]+ Q[t][s] - demand + (1 - delta[t][s]) * M1)     
                    m.addConstr(I[t][s] >= I[t-1][s] + Q[t][s] - demand - (1 - delta[t][s]) * M1)  
                    m.addConstr(I[t-1][s] + Q[t][s] - demand <= delta[t][s]* M1 -0.1) 
                except:
                    print(t)
            m.addConstr(I[t][s] <= delta[t][s] * M1)
            
    # Add chance constraints
    m.addConstr(sum(alpha) <= K)    
    # revise the following constraints to be strong extended formulation
    for t in range(T):
        cum_demand = [sum(i[0:t+1]) for i in sample_detail]
        sort_cum = np.sort(cum_demand)
        index_sort = np.argsort(cum_demand)
        m.addConstr(cash[t][index_sort[K]] >= 0)
        for s in range(K):
            m.addConstr(min_cash[t][index_sort[s]] >= -M2 *beta[t][s])
            m.addConstr(alpha[index_sort[s]] >= beta[t][s])
            if s <= K - 2:
                m.addConstr(beta[t][s] >= beta[t][s+1])    
            
#    # cash constraints
#    for s in range(S):
#        for t in range(T):
#            if t == 0:
#                m.addConstr(ini_cash >= vari_cost * Q[t][s])
#            else:
#                m.addConstr(cash[t-1][s] >= vari_cost * Q[t][s])

    # first-stage decision
    for s in range(S-1):
        m.addConstr(Q[0][s] == Q[0][s+1])
    
    # Set objective
    m.update()
    m.setObjective(expect_cash, GRB.MAXIMIZE)
                       
    # solve
    m.update()
    m.optimize()
    print('') 
    
     # output in txt files
    Qv = [[0 for s in range(S)] for t in range(T)] # ordering quantity in each period for each product
    Iv = [[0 for s in range(S)] for t in range(T)] # end-of-period inventory in each period for each product
    deltav = [[0 for s in range(S)] for t in range(T)] # whether lost-sale not occurs, 0 means occur
    alphav = [0 for s in range(S)]
    betav = [[0 for s in range(K)] for t in range(T)]
    with open('results.txt', 'w') as f:
        f.write('*********************************\n')
        f.write('ordering quantity Q in the first period:\n')
        f.write('%.1f ' % Q[0][0].X)  
        f.write('\n*********************************\n')
        f.write('\n')
        for s in range(S):
            f.write('S%d:\n' % s)
            for t in range(T):
                f.write('%.1f ' % Q[t][s].X)    
                Qv[t][s] = Q[t][s].X
            f.write('\n')
        f.write('\n')
        f.write('***************************************************************************************************************\n')           
        f.write('end-of-period inventory I in each scenario:\n')
        for s in range(S):
            f.write('S%d:\n' % s)
            for t in range(T):
                f.write('%.1f ' % I[t][s].X)    
                Iv[t][s] = I[t][s].X
            f.write('\n')
        f.write('***************************************************************\n')
        f.write('end-of-period cash C in each scenario:\n')
        neg_cash_num = 0
        for s in range(S):
            f.write('S%d:\n' % s)
            record_before = 0
            for t in range(T):
                f.write('%.1f ' % cash[t][s].getValue()) 
                if min_cash[t][s].getValue() < -1 and record_before == 0:
                    neg_cash_num = neg_cash_num + 1
                    record_before = 1
            f.write('\n')
        f.write('*********************************\n')
        f.write('min cash C in each scenario:\n')
        for s in range(S):
            f.write('S%d:\n' % s)
            for t in range(T):
                f.write('%.1f ' % min_cash[t][s].getValue()) 
            f.write('\n')
        f.write('*********************************\n')
        f.write('final expected cash is: %g' % expect_cash.getValue())
        print('final expected cash is: %g' % expect_cash.getValue())
    
    for s in range(S):
        alphav[s] = alpha[s].X
    Q0 = Q[0][0].X
    print('first stage solution: \n')
    print(Q0)
    print('negative scenario number is : %d' % neg_cash_num)
    print('maximum negative scenario number is: %d' % K)
    
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))    
        
except AttributeError:
    print('Encountered an attribute error')
