""" 
# @File  : mip_deterministic_noFinance.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2020/06/30
# @Desc  : the mip model for no finance situation when demand are deterministic

price: mouse 89, headset 159, keyboard 239.
mean demand: 181.54, 397.33, 82.14
vari cost: 70, 130, 200
overhead cost: 20,000

use gurobi in python:
    
    get the value of variable: var.X
    get the value of LinExpr: linExpr.getValue()

"""

import numpy as np
from gurobipy import *
from gurobipy import Model
from gurobipy import GurobiError

# parameter values
ini_cash = 50  # 100000
ini_I = [0, 0, 0]
prices = [20, 5, 15] # [89, 159, 239]
vari_costs = [10, 2, 5] # [70, 130, 200]
overhead_cost = 20 # 20000

T = 3
N = len(ini_I)
delay_length = 0
discount_rate = 0.01

mean_demands = [10, 5, 10] # [181.54, 397.33, 82.14]


try:
    # Create a new model
    m = Model("self-cash-mip")

    # Create variables
    Q = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for n in range(N)] # ordering quantity in each period for each product
    I = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for n in range(N)] # end-of-period inventory in each period for each product
    C = [m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] # LinExpr, end-of-period cash in each period
    w = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for n in range(N)] # lost-sale quantity
    R = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for n in range(N)] # LinExpr, revenue for each product in each period
    
    # objective function
    
    # inventory flow
    for n in range(N):
        for t in range(T):
            if t == 0:
                try:
                    I[n][t] = ini_I[n] + Q[n][t] + w[n][t] - mean_demands[n]
                except:
                    print(n)
                
            else:
                I[n][t] = I[n][t - 1] + Q[n][t] + w[n][t] - mean_demands[n]
                
    # revenue 0
    for n in range(N):
        for t in range(T):
            if t < delay_length:
                R[n][t] = 0
            else:
                R[n][t] = prices[n] * (mean_demands[n] - w[n][t - delay_length])
        
    # cash flow
    for t in range(T):
        if t == 0:
            C[t] = ini_cash + sum([R[n][t] for n in range(N)]) - sum([vari_costs[n] * Q[n][t] for n in range(N)]) - overhead_cost
        else:
            C[t] = C[t - 1] + sum([R[n][t] for n in range(N)]) - sum([vari_costs[n] * Q[n][t] for n in range(N)]) - overhead_cost
            
    discounted_cash = 0
    for n in range(N):
        for k in range(delay_length):
            discounted_cash += R[n][T-1-k] / (1 + discount_rate)**(delay_length - k)    
    final_cash = C[T - 1] + discounted_cash
     
    # Set objective
    m.setObjective(final_cash, GRB.MAXIMIZE)
    
    # Add constraints
    # cash constraint
    for t in range(T):
        if t == 0:
            m.addConstr(ini_cash >= sum([vari_costs[n] * Q[n][t] for n in range(N)]) + overhead_cost) # cash constaints
        else:       
            m.addConstr(C[t - 1] >= sum([vari_costs[n] * Q[n][t] for n in range(N)]) + overhead_cost) # cash constaints
       
    # lost sale constraint
    for n in range(N):
        for t in range(T):
            m.addConstr(w[n][t] <= mean_demands[n])
            
    # non-negavtivety of I_t
    for n in range(N):
        for t in range(T):
            m.addConstr(I[n][t] >= 0)
         
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
       
    print('end-of-period inventory I:')
    for n in range(N):
        print('item %d:' % n)
        for t in range(T):
            print('%.1f' % I[n][t].getValue(), end = ' ')    
        print('')
    print('*********************************')
       
    print('lost-sale quantity w:')
    for n in range(N):
        print('item %d:' % n)
        for t in range(T):
            print('%.1f' % w[n][t].X, end = ' ')    
        print('')  
    print('*********************************')
    
    print('revenue R:')
    for n in range(N):
        print('item %d:' % n)
        for t in range(T):
            print('%.1f' % R[n][t].getValue(), end = ' ')    
        print('')
    print('*********************************')
    
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

