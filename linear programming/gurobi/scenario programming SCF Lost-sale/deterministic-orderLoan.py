# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 10:42:19 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
            deterministic model for order loan
    
"""


from gurobipy import *

# parameter values
ini_cash = 40000
ini_I = [0, 0, 0]
prices = [189, 144, 239]
vari_costs = [140, 80, 150]
overhead_cost = 4000
ini_cash = 25000
B = 10000 # total quantity of order loan
ro = 0.015 # loan rate

T = 6
N = len(ini_I)
delay_length = 2
discount_rate = 0.01
M = 10000

mean_demands = [[46, 77, 38], [338, 389, 144]]


try:
    # Create a new model
    m = Model("Order-loan-mip")

    # Create variables
    Q = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for n in range(N)] # ordering quantity in each period for each product
    I = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for n in range(N)] # end-of-period inventory in each period for each product
    delta = [[m.addVar(vtype = GRB.BINARY) for t in range(T)] for n in range(N)] # whether lost-sale not occurs
    g = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for n in range(N)] # order-loan quantity in each period for each product
    
    C = [LinExpr()  for t in range(T)] # LinExpr, end-of-period cash in each period
    R = [[LinExpr()  for t in range(T + delay_length)] for n in range(N)] # LinExpr, revenue for each product in each period
    
    
    # revenue expression
    for n in range(N):
        for t in range(T + delay_length):
            if t < delay_length:
                R[n][t] = prices[n] * g[n][t]
            elif t == delay_length:
                R[n][t] = prices[n] * (g[n][t]+ini_I[n]+Q[n][t-delay_length]-I[n][t-delay_length]-g[n][t-delay_length]-g[n][t-delay_length]*(1+ro)**delay_length)
            elif t < T:
                R[n][t] = prices[n] * (g[n][t]+I[n][t-delay_length-1]+Q[n][t-delay_length]-I[n][t-delay_length]-g[n][t-delay_length]-g[n][t-delay_length]*(1+ro)**delay_length)
            else:
                R[n][t] = prices[n] * (I[n][t-delay_length-1] + Q[n][t-delay_length]-I[n][t-delay_length]-g[n][t-delay_length]-g[n][t-delay_length]*(1+ro)**delay_length)
                
    
    # cash flow   
    revenue_total = [LinExpr() for t in range(T)]
    vari_costs_total = [LinExpr() for t in range(T)]
    for t in range(T):
        revenue_total[t] = sum([R[n][t] for n in range(N)])
        vari_costs_total[t] = sum([vari_costs[n] * Q[n][t] for n in range(N)])
        if t == 0:
            C[t] = ini_cash + revenue_total[t] - vari_costs_total[t] - overhead_cost
        else:
            C[t] = C[t - 1] + revenue_total[t] - vari_costs_total[t]- overhead_cost
            
    # objective function          
    discounted_cash = LinExpr(0)
    for n in range(N):
        for k in range(delay_length):
            discounted_cash = LinExpr(discounted_cash + R[n][T+k] / (1 + discount_rate)**(k+1))    
    final_cash = LinExpr(C[T - 1] + discounted_cash)
     
    # Set objective
    m.update()
    m.setObjective(final_cash, GRB.MAXIMIZE)
    
    # Add constraints
    # inventory flow    
    for n in range(N):
        for t in range(T):
            if t == 0:
                m.addConstr(I[n][t] <= ini_I[n] + Q[n][t] - mean_demands[n] + (1 - delta[n][t]) * M)     
                m.addConstr(I[n][t] >= ini_I[n] + Q[n][t] - mean_demands[n] - (1 - delta[n][t]) * M)   
                m.addConstr(ini_I[n] + Q[n][t] - mean_demands[n] <= delta[n][t]* M -0.1) 
                m.addConstr(ini_I[n] + Q[n][t] >= mean_demands[n] - (1 -delta[n][t])* M)
            else:
                m.addConstr(I[n][t] <= I[n][t-1]+ Q[n][t] - mean_demands[n] + (1 - delta[n][t]) * M)     
                m.addConstr(I[n][t] >= I[n][t-1] + Q[n][t] - mean_demands[n] - (1 - delta[n][t]) * M)  
                m.addConstr(I[n][t-1] + Q[n][t] - mean_demands[n] <= delta[n][t]* M -0.1) 
                m.addConstr(I[n][t-1] + Q[n][t] >= mean_demands[n] - (1 -delta[n][t])* M) 
            m.addConstr(I[n][t] <= delta[n][t] * M)  
    
    # cash constraint
    for t in range(T):
        if t == 0:
            m.addConstr(ini_cash >= sum([vari_costs[n] * Q[n][t] for n in range(N)]) + overhead_cost) # cash constaints
        else:       
            m.addConstr(C[t - 1] >= sum([vari_costs[n] * Q[n][t] for n in range(N)]) + overhead_cost) # cash constaints      
            
    # non-negavtivety of I_t
    for n in range(N):
        for t in range(T):
            m.addConstr(I[n][t] >= 0)
    
    # order loan quantity less than realized demand
    for n in range(N):
        for t in range(T):
            m.addConstr(g[n][t] <= I[n][t-delay_length-1]+Q[n][t-delay_length]-I[n][t-delay_length])
            
    # total order loan limit
    total_loan = LinExpr()
    for n in range(N):
        for t in range(T):
            total_loan += prices[n] * g[n][t]
    m.addConstr(total_loan <= B)
         
    # solve
    m.optimize()
    print('') 
    
    # output      
    print('*********************************')
    print('ordering quantity Q:')
    for n in range(N):
        print('item %d:' % n)
        for t in range(T):
            print('%.1f' % Q[n][t].X, end = ' ')    
        print('')
    print('*********************************')
       
    print('order loan quantity g:')
    for n in range(N):
        print('item %d:' % n)
        for t in range(T):
            print('%.1f' % g[n][t].X, end = ' ')    
        print('')  
    print('*********************************')
    
    print('end-of-period inventory I:')
    for n in range(N):
        print('item %d:' % n)
        for t in range(T):
            print('%.1f' % I[n][t].X, end = ' ')    
        print('')
    print('*********************************')
    
    print('values of delta:')
    for n in range(N):
        print('item %d:' % n)
        for t in range(T):
            print('%.1f' % delta[n][t].X, end = ' ')    
        print('')
    print('*********************************')
       
    
    print('revenue R:')
    for n in range(N):
        print('item %d:' % n)
        for t in range(T):
            print('%.1f' % R[n][t].getValue(), end = ' ')    
        print('')
    print('*********************************')
    
    print('total revenue in each period:')
    for t in range(T):
        print('%.1f' % revenue_total[t].getValue(), end = ' ')    
    print('\n')
    
    print('total vari costs in each period:')
    for t in range(T):
        print('%.1f' % vari_costs_total[t].getValue(), end = ' ')    
    print('\n')
    
    if not isinstance(discounted_cash, int):
        print('totoal discounted cash: ')
        print('%.1f\n' % discounted_cash.getValue())
    
    print('end-of-period cash C:')
    for t in range(T):
        print('%.1f' % C[t].getValue(), end = ' ')    
    print('\n')
          
    print('Obj: %g' % m.objVal)
    
    
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))
    
except AttributeError:
    print('Encountered an attribute error')


