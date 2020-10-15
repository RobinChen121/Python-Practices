# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 15:35:55 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: build a general scenario model under no finance situation
    
"""


import numpy as np
from gurobipy import *
from gurobipy import Model
from gurobipy import GurobiError
import time
import math


# selected scenario indexs for T=5 : [121, 0, 5, 239, 235, 231, 227, 223, 219, 215]


# Python function to f.write permutations of a given list 
def product(args, repeat):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(args)] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result
           
tic = time.time()
# parameter values
ini_cash = 40000
ini_I = [0, 0, 0]
prices = [90, 160, 300]
vari_costs = [70, 150, 60]
overhead_cost = 2000

T = 5
N = len(ini_I)
delay_length = 1
discount_rate = 0.01
B = 10000 # total quantity of order loan
ro = 0.015 # loan rate


scenario_selected = [121, 0, 5, 52, 158, 16, 113, 107, 190, 59]

demand_scenarios = [[35,191,55], [81,476,331], [222, 918,387]]
demand_possibility = [0.58, 0.22, 0.2]



K = len(demand_possibility) # scenario number in a period
M =  len(scenario_selected)# total selected scenario number
S =  K ** T # total scenario number

scenario_permulations = product(range(K), T)  
scenario_select_detail = [[0 for t in range(T)] for i in range(M)]
index = 0
for i in scenario_selected:
    scenario_select_detail[index] = scenario_permulations[i]
    index = index + 1

        
# set values for scenario links: whether scenario i links with scenario j in period t
scenarioLink = [[[0 for s in range(M)] for s in range(M)] for t in range(T)]
for t in range(T):
    for i in range(M):
        for j in range(M):
            if t == 0:
                if scenario_select_detail[i][t] == scenario_select_detail[j][t]:
                    scenarioLink[t][i][j] = 1
            else:
                if scenarioLink[t-1][i][j] == 1 and \
                        scenario_select_detail[i][t] == scenario_select_detail[j][t]:
                    scenarioLink[t][i][j] = 1
                        
        
# set values for scenario probabilities

scenario_probs = [0 for s in range(M)]
for s in range(M):
    index = scenario_permulations[scenario_selected[s]][0]
    scenario_probs[s] = demand_possibility[index]
    for i in range(1, len(scenario_permulations[scenario_selected[s]])):
        index = scenario_permulations[scenario_selected[s]][i]
        scenario_probs[s] = scenario_probs[s] * demand_possibility[index]
        
scenario_probs_all = [0 for s in range(S)]
for s in range(S):
    index = scenario_permulations[s][0]
    scenario_probs_all[s] = demand_possibility[index]
    for i in range(1, len(scenario_permulations[s])):
        index = scenario_permulations[s][i]
        scenario_probs_all[s] = scenario_probs_all[s] * demand_possibility[index]

# add probability
d = [[0 for s in range(S)] for s in range(S)]
for i in range(S):
    for j in range(i, S):
        for k in range(len(scenario_permulations[0])):
            d[i][j] += (scenario_permulations[i][k] - scenario_permulations[j][k])**2
        d[i][j] = math.sqrt(d[i][j])
        d[j][i] = d[i][j]        


for i in range(S):
    if i not in scenario_selected:
        min_d = 1000000
        min_index = 0
        for j in range(M):
            if d[i][scenario_selected[j]] < min_d:
                min_d = d[i][scenario_selected[j]]
                min_index = j
        scenario_probs[min_index] = scenario_probs[min_index] + scenario_probs_all[i]

                
            
try:
    # Create a new model
    m = Model("self-cash-scenario-mip") 
        
    # Create variables
    # maybe it is better to use LinExpr for I, C and R
    Q = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(M)] for n in range(N)] for t in range(T)] # ordering quantity in each period for each product in each scenario    
    w = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(M)] for n in range(N)] for t in range(T)] # lost-sale quantity in each scenario
    g = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(M)] for n in range(N)] for t in range(T)] # order loan used

        
    I = [[[LinExpr() for s in range(M)] for n in range(N)] for t in range(T)] # end-of-period inventory in each period for each product in each scenario
    C = [[LinExpr() for s in range(M)] for t in range(T)] # LinExpr, end-of-period cash in each period in each scenario
    R = [[[LinExpr() for s in range(M)] for n in range(N)] for t in range(T + delay_length)] # LinExpr, revenue for each product in each period in each scenario

    
    # inventory flow
    for s in range(M):
        for n in range(N):
            for t in range(T):
                index = scenario_permulations[s][t]
                if t == 0:
                    I[t][n][s] = ini_I[n] + Q[t][n][s] + w[t][n][s] - demand_scenarios[index][n]         
                else:
                    I[t][n][s] = I[t-1][n][s] + Q[t][n][s] + w[t][n][s] - demand_scenarios[index][n]
                    
    # revenue
    for s in range(M):
        for n in range(N):
            for t in range(T + delay_length):
                index = scenario_permulations[s][t-delay_length]
                if t < delay_length:
                    R[t][n][s] = prices[n] * g[t][n][s]
                elif t < T:
                    R[t][n][s] = prices[n] * (g[t][n][s]+ demand_scenarios[index][n] - w[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)

                else:
                    R[t][n][s] = prices[n] * (demand_scenarios[index][n] - w[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
    
    # discounted cash           
    discounted_cash = [LinExpr() for s in range(M)]
    for s in range(M):
        for n in range(N):
            for k in range(delay_length):
                discounted_cash[s] += R[T+ k][n][s] / (1 + discount_rate)**(k+1)    
    
    # Add constraints
    
 
    # cash flow
    revenue_total = [[LinExpr() for t in range(T)] for s in range(M)]
    vari_costs_total = [[LinExpr() for t in range(T)] for s in range(M)]
    expect_revenue_total = [LinExpr() for t in range(T)]
    expect_vari_costs_total = [LinExpr() for t in range(T)]
    for s in range(M):
        for t in range(T):
            revenue_total[s][t] = sum([R[t][n][s] for n in range(N)])
            vari_costs_total[s][t] = sum([vari_costs[n] * Q[t][n][s] for n in range(N)])
            if t == 0:
               C[t][s] = ini_cash + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost
            else:
               C[t][s] = C[t-1][s] + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost
    for t in range(T):
        expect_revenue_total[t] = sum([revenue_total[s][t] * scenario_probs[s] for s in range(M)])
        expect_vari_costs_total[t] = sum([vari_costs_total[s][t] * scenario_probs[s] for s in range(M)])
    
    final_expect_cash = sum([scenario_probs[s] * (C[T-1][s] + discounted_cash[s]) for s in range(M)])  
    expect_discounted_cash = sum([scenario_probs[s] * (discounted_cash[s]) for s in range(M)])
    
    # Set objective
    m.update()
    m.setObjective(final_expect_cash, GRB.MAXIMIZE)
    
    # cash constraint
    for s in range(M):
        for t in range(T):
            if t == 0:
                m.addConstr(ini_cash >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost) # cash constaints
            else:       
                m.addConstr(C[t-1][s] >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost) # cash constaints
           
    # lost sale constraint
    for s in range(M):
        for n in range(N):
            for t in range(T):
                index = scenario_permulations[s][t]
                m.addConstr(w[t][n][s] <= demand_scenarios[index][n])
            
    # non-negavtivety of I_t
    for s in range(M):
        for n in range(N):
            for t in range(T):
                m.addConstr(I[t][n][s] >= 0)
                
    # order loan quantity less than realized demand
    for s in range(M):
        for n in range(N):
            for t in range(T):
                index = scenario_permulations[s][t]
                m.addConstr(g[t][n][s] <= demand_scenarios[index][n] - w[t][n][s])
            
    # total order loan limit
    total_loan = [LinExpr() for s in range(M)]
    for s in range(M):
        for n in range(N):
            for t in range(T):
                total_loan[s] += prices[n] * g[t][n][s]
        m.addConstr(total_loan[s] <= B)
         
    
    # non-anticipativity 
    # s1 与 s 的顺序没啥影响       
    # no need for I, R, C      
    for t in range(T):
        for n in range(N):
            for s in range(M):
                m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * Q[t][n][s1] for s1 in range(M)])==\
                    Q[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(M)]))
                m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * w[t][n][s1] for s1 in range(M)])==\
                    w[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(M)]))
                m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * g[t][n][s1] for s1 in range(M)])==\
                    g[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(M)]))
   
     # solve
    m.update()
    #M.Params.Presolve = 0 # can be -1, 0, 1, 2. More aggressive application of presolve takes more time, but can sometimes lead to a significantly tighter model.
    #M.params.Method = 0 # simplex to save memory
    m.optimize()
    
    # output in txt files
    with open('results.txt', 'w') as f:
        f.write('*********************************\n')
        f.write('ordering quantity Q in each scenario:\n')
        for s in range(M):
            f.write('S%d:\n' % s)
            for n in range(N):
                f.write('item %d: ' % n)
                for t in range(T):
                    f.write('%.1f ' % Q[t][n][s].X)    
                f.write('\n')
            f.write('\n')
        f.write('*********************************\n')
        
        f.write('lost-sale quantity w in each scenario:\n')
        for s in range(M):
            f.write('S%d:\n' % s)
            for n in range(N):
                f.write('item %d: ' % n)
                for t in range(T):
                    f.write('%.1f ' % w[t][n][s].X)    
                f.write('\n')
            f.write('\n')
        f.write('*********************************\n')
        
        f.write('order-loan quantity g in each scenario:\n')
        for s in range(M):
            f.write('S%d:\n' % s)
            for n in range(N):
                f.write('item %d: ' % n)
                for t in range(T):
                    f.write('%.1f ' % g[t][n][s].X)    
                f.write('\n')
            f.write('\n')
        f.write('*********************************\n')
        
        f.write('end-of-period inventory I in each scenario:\n')
        for s in range(M):
            f.write('S%d:\n' % s)
            for n in range(N):
                f.write('item %d: ' % n)
                for t in range(T):
                    f.write('%.1f ' % I[t][n][s].getValue())    
                f.write('\n')
            f.write('\n')
        f.write('*********************************\n')       
        
        
        f.write('revenue R in each scenario:\n')
        for s in range(M):
            f.write('S%d:\n' % s)
            for n in range(N):
                f.write('item %d: ' % n)
                for t in range(T):
                    f.write('%.1f ' % R[t][n][s].getValue())    
                f.write('\n')
            f.write('\n')
        f.write('*********************************\n')
        
        
        f.write('discounted cash in each scenario:\n')
        for s in range(M):
            f.write('S%d: ' % s)
            f.write('%.1f ' % discounted_cash[s].getValue())    
            f.write('\n')
        f.write('*********************************\n')
        
        f.write('end-of-period cash C in each scenario:\n')
        for s in range(M):
            f.write('S%d:\n' % s)
            for t in range(T):
                f.write('%.1f ' % C[t][s].getValue()) 
            f.write('  %.3f:\n' % scenario_probs[s])
            f.write('\n')
        f.write('*********************************\n')
        
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
            expect_cash[t] = sum([C[t][s] * scenario_probs[s] for s in range(M)])
            f.write('%.1f ' % expect_cash[t].getValue())           
        f.write('\n')
        f.write('final expected discounted cash is: %g\n' % expect_discounted_cash.getValue())
        f.write('final expected value is: %g' % final_expect_cash.getValue())
    print('final expected value is: %g' % m.objVal)
except GurobiError as e:
    f.write('Error code ' + str(e.errno) + ": " + str(e))    
    
except AttributeError:
    f.write('Encountered an attribute error')

toc = time.time()
time_pass = toc - tic
print('running time is %.2f' % time_pass)
