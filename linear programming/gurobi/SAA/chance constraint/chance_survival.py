# -*- coding: utf-8 -*-
"""
Created on Fri May  7 21:56:07 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description:  A chance-constraint programming approach for cash-flow stochastic inventory flow problems;
              use SAA to solve;
              
              try fluctuant prices;
    
"""


import numpy as np
import scipy.stats as st
from gurobipy import *
import time
from functools import reduce
import itertools
import math


# trunQuantile 似乎在生成随机数上没啥用
def generate_sample(sample_num, trunQuantile, mus):
    T = len(mus)
    samples = [[0 for i in range(sample_num[t])] for t in range(T)]
    for t in range(T):
        for i in range(sample_num[t]):
            rand_p = np.random.uniform(trunQuantile*i/sample_num[t], trunQuantile*(i+1)/sample_num[t])
            samples[t][i] = st.poisson.ppf(rand_p, mus[t])
    return samples
        
  
iniCash = 200
iniI = 0
price  =  [22, 22, 22]
variCostUnit = 10
sal_value = 0.5
mean_demands = [10, 10, 10]
sample_nums = [10, 10, 10]
service_rate = 0.8

T = len(mean_demands)
overheadCost = [150 for t in range(T)]

trunQuantile = 0.9999
S = reduce(lambda x, y: x * y, sample_nums[0:T], 1) # total scenario number

samples = generate_sample(sample_nums, trunQuantile, mean_demands)
# samples = cPickle.load(open("data1.pkl", "rb"))

K = np.floor(S * (1 - service_rate)) # maximum number of scenarios that have negative cash

arr = []
for t in range(T):
    arr.append(range(sample_nums[t]))
scenario_permulations = list(itertools.product(*arr))

tic = time.time()
try:
    # Create a new model
    m = Model("saa-chance")
    
    # Create variables          
    Q = [[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for t in range(T)] # ordering quantity
    delta = [[m.addVar(vtype = GRB.BINARY) for s in range(S)] for t in range(T)] # auxiliary variable
    I = [[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for t in range(T)] # auxiliary variable
    alpha = [[m.addVar(vtype = GRB.BINARY) for s in range(S)] for t in range(T)] # whether cash balance is negative in this scenario
    beta = [m.addVar(vtype = GRB.BINARY) for s in range(S)]
    z = [m.addVar(vtype = GRB.BINARY) for s in range(S)]
    
    cash = [[LinExpr() for s in range(S)] for t in range(T)]
    
    # choose maximum sum of demand in T periods as M1
    M1 = 0
    for s in range(S):
        sCumDemand = 0
        for t in range(T):
            sCumDemand += samples[t][scenario_permulations[s][t]]
        if sCumDemand > M1:
            M1 = sCumDemand
    
    M2 = iniCash + price[0] * M1;
    M3 = variCostUnit * M1 + sum(overheadCost) - iniCash
    
    
    # Set objective
    m.update()
    m.setObjective(sum(z)/S, GRB.MAXIMIZE)
    
    
    # constraints
    # first-stage decision
    for s in range(S-1):
        m.addConstr(Q[0][s] == Q[0][s+1])
     
    # inventory flow
    for s in range(S):
        for t in range(T):
            demand = samples[t][scenario_permulations[s][t]]  # be careful
            if t == 0:
                m.addConstr(I[t][s] <= iniI + Q[t][s] - demand + delta[t][s] * M1)     
                m.addConstr(I[t][s] >= iniI + Q[t][s] - demand - delta[t][s] * M1)   
                m.addConstr(iniI + Q[t][s] - demand <= (1 - delta[t][s]) * M1) 
            else:
                try:
                    m.addConstr(I[t][s] <= I[t-1][s]+ Q[t][s] - demand + delta[t][s] * M1)     
                    m.addConstr(I[t][s] >= I[t-1][s] + Q[t][s] - demand - delta[t][s] * M1)  
                    m.addConstr(I[t-1][s] + Q[t][s] - demand <= (1 - delta[t][s]) * M1) 
                except:
                    print(t)
            m.addConstr(I[t][s] <= (1 - delta[t][s]) * M1)
        
    # cash flow
    for t in range(T):
        for s in range(S):
            if t == 0 and T > 1: # end-of-period cash balance in the first period
                cash[t][s] = iniCash + price[t] * (iniI + Q[t][s] - I[t][s]) - variCostUnit * Q[t][s] - overheadCost[t]
            elif t == 0 and T <= 1:
                cash[t][s] = iniCash + price[t] * (iniI + Q[t][s] - I[t][s]) - variCostUnit * Q[t][s] - overheadCost[t] + sal_value * I[t][s]
            elif t == T - 1 and T > 1:
                cash[t][s] = cash[t-1][s] + price[t] * (I[t-1][s] + Q[t][s] - I[t][s]) - variCostUnit * Q[t][s] - overheadCost[t] + sal_value * I[t][s]
            else:
                cash[t][s] = cash[t-1][s] + price[t] * (I[t-1][s] + Q[t][s] - I[t][s]) - variCostUnit * Q[t][s] - overheadCost[t]

    m.update()    
    
    # the relation of cash balance with alpha
    for s in range(S):
        for t in range(T):
            m.addConstr(cash[t][s] <= M2 * alpha[t][s])
            m.addConstr(cash[t][s] >= -M3 * (1 - alpha[t][s]))    
    for s in range(S):
        sumAlpha = LinExpr()
        for t in range(T):
            sumAlpha = sumAlpha + alpha[t][s]
            m.addConstr(z[s] <= alpha[t][s])
        m.addConstr(1 - z[s] <= T - sumAlpha)
        m.addConstr(delta[t][s] <= beta[s])

            
    # Add chance constraints
    # careful, should revise
    m.addConstr(sum(beta) <= math.floor(S* (1 - service_rate)))  
                       
    # solve
    m.update()
    m.optimize()
    print('') 
    
     # output in txt files
    Qv = [[0 for s in range(S)] for t in range(T)] # ordering quantity in each period for each product
    Iv = [[0 for s in range(S)] for t in range(T)] # end-of-period inventory in each period for each product
    deltav = [[0 for s in range(S)] for t in range(T)] # whether lost-sale occurs, 1 means occur
    alphav = [[0 for s in range(S)] for t in range(T)]
    betav = [0 for s in range(S)]
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
                deltav[t][s] = delta[t][s].X
                alphav[t][s] = alpha[t][s].X
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
        for s in range(S):
            f.write('S%d:\n' % s)
            for t in range(T):
                f.write('%.1f ' % cash[t][s].getValue()) 
            f.write('\n')
        f.write('*********************************\n')
    
    for s in range(S):
        betav[s] = beta[s].X
    Q0 = Q[0][0].X
    print('first stage solution Q0 = {:.2f} '.format(Q0))
    print('maximum survival probability is {:.5%}'.format(m.objVal))
    print('total scenario number is : %d' % S)
    print('lost sale scenario number is: %d' % sum(betav))
    lostRate = sum(betav) / S
    print('lost sale rate is {:%} '.format(lostRate))
    
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))    
        
except AttributeError:
    print('Encountered an attribute error')
    


print('\n')
toc = time.time()
time_pass = toc - tic
print('running time is %.2f' % time_pass)
print('final expected value is: %g\n' % m.objVal)
    