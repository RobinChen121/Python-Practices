# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:16:10 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description:  saa modeling for the no financing siutation；

6 periods, 3 samples in each period, running time is 46s, value = 45171;
5 periods, 5 samples in each period, running time is 129.67s;
6 periods, 5 samples in the first 5 periods and 3 samples in the final:
     843766 rows, 562518 columns, 3724009 nonzeros, runnding time is 1862.54s; value = 45348;
     scenario tree value is 45188 for 3 scenarios in each period;
4 periods, 5 samples in each period, running time is 25s, value = 33417;
    scenario tree value is 33999 for 5 scenarios in each period, running time 49s;
    scenario tree value is 33775 for 3 scenarios in each period, running time 1s;
     

end-of-period cash can be negative in the last period because paying back order-loan
    
"""

from gurobipy import *
from gurobipy import LinExpr
from gurobipy import GRB
from gurobipy import Model
import time
import numpy as np
import scipy.stats as st
from math import exp
import itertools


def lognorm_ppf(x, mu, sigma):
    shape  = sigma
    loc    = 0
    scale  = exp(mu)
    return st.lognorm.ppf(x, shape, loc, scale)

def generate_sample(sample_num, trunQuantile, mus, sigmas, booming_demand):
    T = len(booming_demand)
    N = len(mus)
    samples = [[[0 for i in range(sample_num[t])] for n in range(N)] for t in range(T)]
    for t in range(T):
        for i in range(sample_num[t]):
            rand_p = np.random.uniform(trunQuantile*i/sample_num[t], trunQuantile*(i+1)/sample_num[t])
            for n in range(N):
                samples[t][n][i] = lognorm_ppf(rand_p, mus[n][booming_demand[t]], sigmas[n][booming_demand[t]])
    return samples

  
    
# parameter values
ini_I = [0, 0, 0]
# prices = [89, 159, 300]
# vari_costs = [70, 60, 60]
prices = [189, 144, 239]
vari_costs = [140, 70, 150]
ini_cash = 20000

T = 2
overhead_cost = [2000 for t in range(T)]
booming_demand = [0, 0, 0, 0, 1, 1]
delay_length = 0
discount_rate = 0.01
B = 10000  # total quantity of order loan
ro = 0.015  # loan rate
M = 10000

mus = [[3.66, 5.79], [4.13, 5.91], [3.54, 4.96]]
sigmas = [[0.6, 0.26], [0.66, 0.33], [0.46, 0.18]]
#mus = [[3.66, 5.79], [4.13, 5.91]]
#sigmas = [[0.6, 0.26], [0.66, 0.33]]
N = len(mus)
sample_nums = [5, 5, 5, 5, 3, 3]
trunQuantile = 0.999

samples = generate_sample(sample_nums, trunQuantile, mus, sigmas, booming_demand[0:T])
S = np.prod(sample_nums[0:T])
arr = []
for t in range(T):
    arr.append(range(sample_nums[t]))
scenario_permulations = list(itertools.product(*arr))

tic = time.time()
try:
    # Create a new model
    m = Model("order-loan-saa")

    # Create variables
#    Q0 = [m.addVar(vtype = GRB.CONTINUOUS) for n in range(N)]
    Q = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)]for n in range(N)] for t in range(T)]
    I = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for t in range(N)] for n in range(T)] # end-of-period inventory in each period for each product
    delta = [[[m.addVar(vtype = GRB.BINARY) for s in range(S)] for t in range(N)] for n in range(T)] # whether lost-sale not occurs
    g = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for n in range(N)] for t in range(T)] # order-loan quantity in each period for each product
     
    C = [[LinExpr()  for s in range(S)] for t in range(T)] # LinExpr, end-of-period cash in each period
    R = [[[LinExpr()  for s in range(S)] for n in range(N)] for t in range(T + delay_length)]  # LinExpr, revenue for each product in each period
    
    # revenue expression  # check revenue
    for s in range(S):
        for n in range(N):
            for t in range(T + delay_length):
                if t < delay_length:
                    R[t][n][s] = prices[n] * g[t][n][s]
                else:
                    if t == delay_length:
                        R[t][n][s] = prices[n]*(ini_I[n]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
                    elif t < T:
                        R[t][n][s] = prices[n]*(g[t][n][s]+I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
                    else:        
                        R[t][n][s] = prices[n]*(I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
    
    m.update()
    
    # cash flow   
    revenue_total = [[LinExpr() for s in range(S)] for t in range(T)]
    vari_costs_total = [[LinExpr() for s in range(S)] for t in range(T)]
    expect_revenue_total = [LinExpr() for t in range(T)]
    expect_vari_costs_total = [LinExpr() for t in range(T)]
    for s in range(S):
        for t in range(T):
            revenue_total[t][s] = sum([R[t][n][s] for n in range(N)])
            vari_costs_total[t][s] = sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) # strange
            try:
                if t == 0:
                    C[t][s] = ini_cash + revenue_total[t][s] - vari_costs_total[t][s] - overhead_cost[t]
                else:
                    C[t][s] = C[t-1][s] + revenue_total[t][s] - vari_costs_total[t][s]- overhead_cost[t]
            except:
                print(n)   
    
    for t in range(T):
        expect_revenue_total[t] = sum([revenue_total[t][s] / S for s in range(S)])
        expect_vari_costs_total[t] = sum([vari_costs_total[t][s] / S for s in range(S)])
        
        
    m.update()
            
    # objective function          
    discounted_cash = [LinExpr() for s in range(S)]
    for s in range(S):
        for n in range(N):
            for k in range(delay_length):
                discounted_cash[s] = discounted_cash[s] + R[T+k][n][s] / (1 + discount_rate)**(k+1)    
    final_cash = sum([(C[T-1][s] + discounted_cash[s])/ S for s in range(S)])
    expect_discounted_cash = sum([(discounted_cash[s])/ S for s in range(S)])
    
    
    # Add constraints
#    for s in range(S):
#        for n in range(N):
#            for t in range(T):
#                m.addConstr(Q0 == 0) 
    # inventory flow   
    for s in range(S):
        for n in range(N):
            for t in range(T):
                demand = samples[t][n][scenario_permulations[s][t]]  # be careful
                if t == 0:
                    m.addConstr(I[t][n][s] <= ini_I[n] + Q[t][n][s] - demand + (1 - delta[t][n][s]) * M)     
                    m.addConstr(I[t][n][s] >= ini_I[n] + Q[t][n][s] - demand - (1 - delta[t][n][s]) * M)   
                    m.addConstr(ini_I[n] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                else:
                    try:
                        m.addConstr(I[t][n][s] <= I[t-1][n][s]+ Q[t][n][s] - demand  + (1 - delta[t][n][s]) * M)     
                        m.addConstr(I[t][n][s] >= I[t-1][n][s] + Q[t][n][s] - demand  - (1 - delta[t][n][s]) * M)  
                        m.addConstr(I[t-1][n][s] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                    except:
                        print(n)
                m.addConstr(I[t][n][s] <= delta[t][n][s] * M)         
        
    # cash constraint
    for s in range(S):
        for t in range(T):
            if t == 0:
                m.addConstr(ini_cash  >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t]) # cash constaints
            else:       
                m.addConstr(C[t - 1][s]  >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t]) # cash constaints  
    
    # non-negavtivety of I_t
    for s in range(S):
        for n in range(N):
            for t in range(T):
                m.addConstr(I[t][n][s] >= 0)
    
     # order loan quantity less than realized demand
    for s in range(S):
        for n in range(N):
            for t in range(T):
                m.addConstr(g[t][n][s] <= I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s])
            
    # total order loan limit
    total_loan = [LinExpr() for s in range(S)]
    for s in range(S):
        for n in range(N):
            for t in range(T):
                total_loan[s] += prices[n] * g[t][n][s]
    for s in range(S):
        m.addConstr(total_loan[s] <= B)
        
    # first-stage decision
    for s in range(S-1):
        for n in range(N):
            m.addConstr(Q[0][n][s] == Q[0][n][s+1])
                
    
                    
    # Set objective
    m.update()
    m.setObjective(final_cash, GRB.MAXIMIZE)
                       
    # solve
    m.update()
    m.optimize()
    print('') 
    
    # output in txt files
    Qv = [[[0 for s in range(S)] for n in range(N)] for t in range(T)] # ordering quantity in each period for each product
    Iv = [[[0 for s in range(S)] for n in range(N)] for t in range(T)] # end-of-period inventory in each period for each product
    deltav = [[[0 for s in range(S)] for n in range(N)] for t in range(T)] # whether lost-sale not occurs, 0 means occur
    gv = [[[0 for s in range(S)] for n in range(N)] for t in range(T)] # order-loan quantity in each period for each product
    with open('results.txt', 'w') as f:
        f.write('*********************************\n')
        f.write('ordering quantity Q in the first period:\n')
        for n in range(N):
            f.write('item %d: ' % n)
            f.write('%.1f ' % Q[0][n][0].X)  
        print('ordering quantity Q in the first period:\n')
        for n in range(N):
            print('item %d: ' % n)
            print('%.1f ' % Q[0][n][0].X) 
        f.write('\n*********************************\n')
        f.write('ordering quantity Q in each scenario:\n')
        for s in range(S):
            f.write('S%d:\n' % s)
            for n in range(N):
                f.write('item %d: ' % n)
                for t in range(T):
                    f.write('%.1f ' % Q[t][n][s].X)    
                    Qv[t][n][s] = Q[t][n][s].X
                f.write('\n')
            f.write('\n')
        f.write('***************************************************************************************************************\n')           
        f.write('order loan g used in each scenario:\n')
        count = 0
        for s in range(S):
            count_flag = 0
            f.write('S%d:\n' % s)
            for n in range(N):
                f.write('item %d: ' % n)
                for t in range(T):
                    f.write('%.1f ' % g[t][n][s].X)    
                    if g[t][n][s].X > 0.5:
                        count_flag = 1
                    gv[t][n][s] = g[t][n][s].X
                f.write('\n')
            if count_flag == 1:
                count = count + 1
            f.write('\n')
        f.write('times of order loan used in total %d ' % count)
        print('times of order loan used in total %d ' % count)
        percent = count / (3**T)
        print('order loan used percent %.4f%% ' % percent)
        f.write('\n************************\n')
        f.write('end-of-period inventory I in each scenario:\n')
        for s in range(S):
            f.write('S%d:\n' % s)
            for n in range(N):
                f.write('item %d: ' % n)
                for t in range(T):
                    f.write('%.1f ' % I[t][n][s].X)    
                    Iv[t][n][s] = I[t][n][s].X
                f.write('\n')
            f.write('\n')
        f.write('***************************************************************\n')
        
        f.write('not lost-sale delta in each scenario:\n')
        for s in range(S):
            f.write('S%d:\n' % s)
            for n in range(N):
                f.write('item %d: ' % n)
                for t in range(T):
                    f.write('%.1f ' % delta[t][n][s].X)  
                    deltav[t][n][s] = delta[t][n][s].X
                f.write('\n')
            f.write('\n')
        f.write('*********************************\n')
        
        
        f.write('revenue R in each scenario:\n')
        for s in range(S):
            f.write('S%d:\n' % s)
            for n in range(N):
                f.write('item %d: ' % n)
                for t in range(T):
                    f.write('%.1f ' % R[t][n][s].getValue())    
                f.write('\n')
            f.write('\n')
        f.write('*********************************\n')
        
        f.write('end-of-period cash C in each scenario:\n')
        for s in range(S):
            f.write('S%d:\n' % s)
            for t in range(T):
                f.write('%.1f ' % C[t][s].getValue()) 
            f.write('\n')
        f.write('*********************************\n')
        
        f.write('discounted cash in each scenario:\n')
        for s in range(S):
            f.write('S%d: ' % s)
            f.write('%.1f ' % discounted_cash[s].getValue())    
            f.write('\n')
        f.write('\n*********************************\n')
            
        f.write('expectd Revenue in each period:\n')
        for t in range(T):
            f.write('%.1f ' % expect_revenue_total[t].getValue()) 
        f.write('\n')
        f.write('varicosts in each period:\n')
        for t in range(T):
            f.write('%.1f ' % expect_vari_costs_total[t].getValue()) 
        f.write('\n')
        f.write('expected end-of-period cash in each period:\n')
        f.write('%.1f ' % ini_cash) 
        expect_cash = [LinExpr() for t in range(T)]
        for t in range(T):
            expect_cash[t] = sum([C[t][s] / S for s in range(S)])
            f.write('%.1f ' % expect_cash[t].getValue())           
        f.write('\n')
        f.write('final expected discounted cash is: %g\n' % expect_discounted_cash.getValue())
        f.write('final expected cash is: %g' % final_cash.getValue())
    print('final expected value is: %g' % m.objVal)
    
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))    
        
except AttributeError:
    print('Encountered an attribute error')
    
toc = time.time()
time_pass = toc - tic
print('running time is %.2f' % time_pass)





