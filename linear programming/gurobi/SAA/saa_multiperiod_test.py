# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:21:12 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description:  test SAA method for a multi period stochastic inventory problem,
              no inventory holding cost;
              more samples may have bigger upper bounds;

gurobi functions: addGenConstrMax()
    
"""

import numpy as np
import scipy.stats as st
from gurobipy import *
import time
from functools import reduce
import itertools


def generate_sample(sample_num, trunQuantile, mu):
    samples = [0 for i in range(sample_num)]
    for i in range(sample_num):
        rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
        samples[i] = st.poisson.ppf(rand_p, mu)
    return samples
        
  
ini_cash = 10
price  =  6
vari_cost = 2
sal_value = 1
mean_demands = [5, 5, 5, 5, 5, 5]
T = len(mean_demands)
sample_nums = [5, 3, 3, 3, 5, 5]
trunQuantile = 0.999
scenario_num = reduce(lambda x, y: x * y, sample_nums, 1)

samples_total = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    samples_total[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
scenarios = list(itertools.product(*samples_total))    

tic = time.time()
try:
    # Create a new model
    m = Model("saa-mip")
    
    # Create variables
    Q = [[] for t in range(T)]
    for t in range(T):
        if t == 0:
            Q[t].append(m.addVar(vtype = GRB.CONTINUOUS)) # first-stage decision, only one value for all scenarios
        else:
            Q[t].extend([m.addVar(vtype = GRB.CONTINUOUS) for s in range(scenario_num)])
            
    Rd = [[m.addVar(vtype = GRB.CONTINUOUS) for s in range(scenario_num)] for t in range(T)] # realized demand, auxiliary variable
    delta = [[m.addVar(vtype = GRB.BINARY) for s in range(scenario_num)] for t in range(T)] # realized demand, auxiliary variable
    Iplus = [[m.addVar(vtype = GRB.CONTINUOUS) for s in range(scenario_num)] for t in range(T)] # auxiliary variable
    profit = [[LinExpr() for s in range(scenario_num)] for t in range(T)]
    total_expect_profit = LinExpr()
    M = 100000

    for t in range(T):
        for s in range(scenario_num):
            if t == 0 and T > 1:
                profit[t][s] = price * Rd[t][s] - vari_cost * Q[t][0]
            elif t == 0 and T <= 1:
                profit[t][s] = price * Rd[t][s] - vari_cost * Q[t][0] + sal_value * Iplus[t][s]
            elif t == T - 1 and T > 1:
                profit[t][s] = profit[t-1][s] + price * Rd[t][s]  - vari_cost * Q[t][s] + sal_value * Iplus[t][s]
            else:
                profit[t][s] = profit[t-1][s] + price * Rd[t][s]  - vari_cost * Q[t][s]
        
    
    total_expect_profit = sum(profit[T - 1]) / scenario_num
    m.update()
    m.setObjective(total_expect_profit, GRB.MAXIMIZE)
    
    
    # Add constraints
    for t in range(T):
        for s in range(scenario_num):
            m.addConstr(Rd[t][s] >= scenarios[s][t] - delta[t][s] * M)
            m.addConstr(Rd[t][s] <= scenarios[s][t])
            if t == 0:
                m.addConstr(Rd[t][s] <= Q[t][0] + (1 - delta[t][s]) * M)
                m.addConstr(Rd[t][s] >= Q[t][0] - (1 - delta[t][s]) * M)
                m.addConstr(Q[t][0] >= scenarios[s][t] - delta[t][s] * M)
                m.addConstr(Iplus[t][s] == Q[t][0] - Rd[t][s])
            else:
                m.addConstr(Rd[t][s] <= Iplus[t-1][s] + Q[t][s] + (1 - delta[t][s]) * M)
                m.addConstr(Rd[t][s] >= Iplus[t-1][s] + Q[t][s] - (1 - delta[t][s]) * M)
                m.addConstr(Iplus[t-1][s] + Q[t][s] >= scenarios[s][t] - delta[t][s] * M)
                m.addConstr(Iplus[t][s] == Iplus[t-1][s] + Q[t][s] - Rd[t][s])
                
    # Add cash constraints
    for t in range(T):
        for s in range(scenario_num):
            if t == 0:
                m.addConstr( vari_cost * Q[t][0] <= ini_cash)
            else:
                m.addConstr(vari_cost * Q[t][s] <= profit[t-1][s])
    
    
    
    m.update()
    m.optimize()
    
    toc = time.time()
    time_pass = toc - tic 
    print('ordering quantity in the first period is:  %.2f ' % Q[0][0].X)
    print('final expected profit is:  %.2f ' % total_expect_profit.getValue())
    print('running time is %.2f s' % time_pass)
    
    # output results in txt file
    with open('results.txt', 'w') as f:
        f.write('demand samples in each period---------\n')
        for t in range(T):
            period  = t + 1
            f.write('period %d: ' % period)
            for s in range(scenario_num):
                f.write('%.2f ' % scenarios[s][t])
            f.write('\n')
        f.write('\n')
        f.write('end-of-period inventory in each period---------\n')
        for t in range(T):
            period  = t + 1
            f.write('period %d: ' % period)
            for s in range(scenario_num):
                f.write('%.2f ' % Iplus[t][s].X)
            f.write('\n')
        f.write('\n')
        f.write('end-of-period profit in each period---------\n')
        for t in range(T):
            period  = t + 1
            f.write('period %d: ' % period)
            for s in range(scenario_num):
                f.write('%.2f ' % profit[t][s].getValue())
            f.write('\n')
        f.write('\n')
    
except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))    
        
except AttributeError:
        print('Encountered an attribute error')
        
