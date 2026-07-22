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
from memory_profiler import profile

# Python function to f.write permutations of a given list 
def product(args, repeat):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(args)] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result

#@profile   
def lp():       
    tic = time.time()
    # parameter values
    ini_cash = 20000
    ini_I = [0, 0, 0]
    prices = [189, 144, 239]
    vari_costs = [140, 80, 150]
    overhead_cost = 5000
    
    T = 3
    N = len(ini_I)
    delay_length = 2
    discount_rate = 0.01
    
    #demand_scenarios = [[25,290,109], [58,365,90], [62, 134, 132], [289, 789, 273],\
    #                    [74, 965, 564]]
    #demand_possibility = [0.105, 0.341, 0.33, 0.106, 0.119]
    
#    demand_scenarios = [[76,70,135], [38,303,65], [68, 681, 236], [292, 898, 584]]
#    demand_possibility = [0.278, 0.359, 0.262, 0.101]
    
    #demand_scenarios = [[35,290,109], [58,365,90]]
    #demand_possibility = [0.5, 0.5]
    
    
    demand_scenarios = [[27,129,39], [56,235,57], [39, 57,26]]
    demand_possibility = [0.284, 0.116, 0.6]
    
    K = len(demand_possibility) # scenario number in a period
    S =  K ** T # total scenario number
    
    # set values for scenario links: whether scenario i links with scenario j in period t
    scenarioLink = [[[0 for s in range(S)] for s in range(S)] for t in range(T)]
    for t in range(T):
        slices = round(S * (1 / K)**(t+1)) # number of scenario in a slice
        slice_num = round(K**(t+1))       # totoal number of slices
        for i in range(slice_num):
            for j in range(slices * i, slices * (i + 1)):
                for k in range(slices * i, slices * (i + 1)):
                    scenarioLink[t][j][k] = 1
            
    # set values for scenario probabilities
    scenario_permulations = product(range(K), T)  
    scenario_probs = [0 for s in range(S)]
    for s in range(S):
        index = scenario_permulations[s][0]
        scenario_probs[s] = demand_possibility[index]
        for i in range(1, len(scenario_permulations[s])):
            index = scenario_permulations[s][i]
            scenario_probs[s] = scenario_probs[s] * demand_possibility[index]
    
    try:
        # Create a new model
        m = Model("self-cash-scenario-mip") 
            
        # Create variables
        # maybe it is better to use LinExpr for I, C and R
        Q = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for n in range(N)] for t in range(T)] # ordering quantity in each period for each product in each scenario    
        w = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for n in range(N)] for t in range(T)] # lost-sale quantity in each scenario
                
        I = [[[LinExpr() for s in range(S)] for n in range(N)] for t in range(T)] # end-of-period inventory in each period for each product in each scenario
        C = [[LinExpr() for s in range(S)] for t in range(T)] # LinExpr, end-of-period cash in each period in each scenario
        R = [[[LinExpr() for s in range(S)] for n in range(N)] for t in range(T)] # LinExpr, revenue for each product in each period in each scenario
        
        # revenue
        for s in range(S):
            for n in range(N): 
                for t in range(T):                         
                    index = scenario_permulations[s][t]
                    R[t][n][s] = prices[n] * (demand_scenarios[index][n] - w[t][n][s])
        
        # discounted cash           
        discounted_cash = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for k in range(delay_length):
                    discounted_cash[s] += R[T-k-1][n][s] / (1 + discount_rate)**(k+1)    
    
        
        # Add constraints
        # inventory flow
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    index = scenario_permulations[s][t]
                    if t == 0:
                        I[t][n][s] = ini_I[n] + Q[t][n][s] + w[t][n][s] - demand_scenarios[index][n]          
                    else:
                        I[t][n][s] = I[t-1][n][s] + Q[t][n][s] + w[t][n][s] - demand_scenarios[index][n]
     
        # cash flow
        revenue_total = [[LinExpr() for t in range(T)] for s in range(S)]
        vari_costs_total = [[LinExpr() for t in range(T)] for s in range(S)]
        expect_revenue_total = [LinExpr() for t in range(T)]
        expect_vari_costs_total = [LinExpr() for t in range(T)]
        for s in range(S):
            for t in range(T):
                revenue_total[s][t] = sum([R[t][n][s] for n in range(N)])
                vari_costs_total[s][t] = sum([vari_costs[n] * Q[t][n][s] for n in range(N)])
                if t == 0:
                    if t == delay_length:
                        C[t][s] = ini_cash + revenue_total[s][t] - vari_costs_total[s][t] - overhead_cost
                    else:
                        C[t][s] = ini_cash - vari_costs_total[s][t] - overhead_cost
                else:
                    if t < delay_length:
                        C[t][s] = C[t-1][s] - vari_costs_total[s][t] - overhead_cost
                    else:
                        C[t][s] = C[t-1][s] + revenue_total[s][t-delay_length]- vari_costs_total[s][t] - overhead_cost
        for t in range(T):
            expect_revenue_total[t] = sum([revenue_total[s][t] * scenario_probs[s] for s in range(S)])
            expect_vari_costs_total[t] = sum([vari_costs_total[s][t] * scenario_probs[s] for s in range(S)])
        
        # Set objective
        final_expect_cash = sum([scenario_probs[s] * (C[T-1][s] + discounted_cash[s]) for s in range(S)])  
        expect_discounted_cash = sum([scenario_probs[s] * (discounted_cash[s]) for s in range(S)])
        m.update()
        m.setObjective(final_expect_cash, GRB.MAXIMIZE)
        
        # cash constraint
        for s in range(S):
            for t in range(T):
                if t == 0:
                    m.addConstr(ini_cash >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost) # cash constaints
                else:       
                    m.addConstr(C[t-1][s] >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost) # cash constaints
               
        # lost sale constraint
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    index = scenario_permulations[s][t]
                    m.addConstr(w[t][n][s] <= demand_scenarios[index][n])
                
        # non-negavtivety of I_t
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    m.addConstr(I[t][n][s] >= 0)
        
        # non-anticipativity 
        # s1 与 s 的顺序没啥影响       
        # no need for I, R, C      
        for t in range(T):
            for n in range(N):
                for s in range(S):
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * Q[t][n][s1] for s1 in range(S)])==\
                        Q[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
                    m.addConstr(sum([scenarioLink[t][s1][s] * scenario_probs[s1] * w[t][n][s1] for s1 in range(S)])==\
                        w[t][n][s] * sum([scenarioLink[t][s1][s] * scenario_probs[s1] for s1 in range(S)]))
    
        
         # solve
        m.update()
        #m.Params.Presolve = 0 # can be -1, 0, 1, 2. More aggressive application of presolve takes more time, but can sometimes lead to a significantly tighter model.
        #m.params.Method = 0 # simplex to save memory
#        m.write('scenario_noFinance.mps')
#        m.write('scenario_noFinance.lp')
        #m.params.Threads = 1
        
        m.optimize()
        
        # output in txt files
        with open('results.txt', 'w') as f:
            f.write('*********************************\n')
            f.write('ordering quantity Q in each scenario:\n')
            for s in range(S):
                f.write('S%d:\n' % s)
                for n in range(N):
                    f.write('item %d: ' % n)
                    for t in range(T):
                        f.write('%.1f ' % Q[t][n][s].X)    
                    f.write('\n')
                f.write('\n')
            f.write('*********************************\n')
            
            f.write('end-of-period inventory I in each scenario:\n')
            for s in range(S):
                f.write('S%d:\n' % s)
                for n in range(N):
                    f.write('item %d: ' % n)
                    for t in range(T):
                        f.write('%.1f ' % I[t][n][s].getValue())    
                    f.write('\n')
                f.write('\n')
            f.write('*********************************\n')
            
            f.write('lost-sale quantity w in each scenario:\n')
            for s in range(S):
                f.write('S%d:\n' % s)
                for n in range(N):
                    f.write('item %d: ' % n)
                    for t in range(T):
                        f.write('%.1f ' % w[t][n][s].X)    
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
            
            
            f.write('discounted cash in each scenario:\n')
            for s in range(S):
                f.write('S%d: ' % s)
                f.write('%.1f ' % discounted_cash[s].getValue())    
                f.write('\n')
            f.write('*********************************\n')
            
            f.write('end-of-period cash C in each scenario:\n')
            for s in range(S):
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
                expect_cash[t] = sum([C[t][s] * scenario_probs[s] for s in range(S)])
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
    
lp()
