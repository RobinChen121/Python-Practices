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

from gurobipy import *

# parameter values
ini_cash = 10000
ini_I = [0, 0, 0]
prices = [90, 160, 200]
vari_costs = [50, 120, 140]
overhead_cost = 2000

T = 3
N = len(ini_I)
delay_length = 1
discount_rate = 0.01
M = 10000

mean_demands = [100, 200, 100]


try:
    # Create a new model
    m = Model("self-cash-mip")

    # Create variables
    Q = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for n in range(N)] # ordering quantity in each period for each product
    I = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for n in range(N)] # end-of-period inventory in each period for each product
    C = [m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] # LinExpr, end-of-period cash in each period
    w = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T)] for n in range(N)] # lost-sale quantity
    R = [[m.addVar(vtype = GRB.CONTINUOUS) for t in range(T + delay_length)] for n in range(N)] # LinExpr, revenue for each product in each period
    
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
                m.addConstr(I[n][t] == ini_I[n] + Q[n][t] + w[n][t] - mean_demands[n])          
            else:
                m.addConstr(I[n][t] == I[n][t - 1] + Q[n][t] + w[n][t] - mean_demands[n])
    
    # revenue 
    for n in range(N):
        for t in range(T + delay_length):
            if t < delay_length:
                m.addConstr(R[n][t] == 0)
            else:
                m.addConstr(R[n][t] == prices[n] * (mean_demands[n] - w[n][t-delay_length]))
    
    # cash flow   
    revenue_total = [LinExpr() for t in range(T)]
    vari_costs_total = [LinExpr() for t in range(T)]
    for t in range(T):
        revenue_total[t] = sum([R[n][t] for n in range(N)])
        vari_costs_total[t] = sum([vari_costs[n] * Q[n][t] for n in range(N)])
        if t == 0:
            m.addConstr(C[t] == ini_cash + revenue_total[t] - vari_costs_total[t] - overhead_cost)
        else:
            m.addConstr(C[t] == C[t - 1] + revenue_total[t] - vari_costs_total[t]- overhead_cost)
  
    
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
            print('%.1f' % I[n][t].X, end = ' ')    
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
            print('%.1f' % R[n][t].X, end = ' ')    
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
        print('%.1f' % C[t].X, end = ' ')    
    print('\n')
          
    print('Obj: %g' % m.objVal)
    
    
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))
    
except AttributeError:
    print('Encountered an attribute error')

