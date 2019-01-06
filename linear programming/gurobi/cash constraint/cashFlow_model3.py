# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 11:39:08 2019

@author: Zhen Chen

@Python version: 3.6

@description: this is for testing a numercial example by adopting Gurobi to solve
              the linear model. 
              The backgroud of this problem is about a small online retailer.
              In the model3, there exits fixed ordering cost and order-up-to level is 
              a decision variable.
    
"""


from gurobipy import *
import numpy as np
import time


iniInventory = 0 # initial inventory
iniCash = 300 # initial cash
meanDemand = [20, 40, 60, 40] # mean demand in each week
T = len(meanDemand) # length of planning horizon
M = 10000

price = np.ones(T) * 50
variCost = np.ones(T) * 20 # unit purchasing cost from the supplier
holdCost = 1
K = np.ones(T) * 0 # overhead costs for each period
G = np.ones(T) * 100 # fixed ordering cost for each period
alpha = 0 # revenue sharing rate


try:
    m = Model("mip_cash_constraint_model3")
    
    # decision variables
    I = {}
    S = {}
    x = {}
    for i in range(T):
        I_name = 'I' + str(i)
        S_name = 'S' + str(i)
        x_name = 'x' + str(i)
        I[i] = m.addVar(vtype = GRB.CONTINUOUS, name = I_name)
        S[i] = m.addVar(vtype = GRB.CONTINUOUS, name = S_name)
        x[i] = m.addVar(vtype = GRB.BINARY, name = x_name)
    
    
    # objective function
    cash = {}
    cash[0] = iniCash + (1-alpha) * price[0] * (S[0] - I[0]) - variCost[0] * (S[0] - iniInventory)\
               - holdCost * I[0] - K[0] - G[0] * x[0]
    for i in range(1, T):
       cash[i] = cash[i - 1]+ (1-alpha) * price[i] * (S[i] - I[i]) - variCost[i] * (S[i] - I[i - 1])\
                  - holdCost * I[i] - K[i] - G[i] * x[i]
    
    m.setObjective(cash[T - 1], GRB.MAXIMIZE)
    
    
    # constraints
    m.addConstr(variCost[0] * (S[0] - iniInventory) + G[0] * x[0] <= iniCash)
    m.addConstr(S[0] - iniInventory <= M * x[0])
    for t in range(1, T):
        m.addConstr(variCost[t] * (S[t] - I[t - 1]) + G[t] * x[t] <= cash[t - 1]) 
        m.addConstr(S[t] - I[t - 1] <= M * x[t])
    
    m.addConstr(iniInventory <= S[0])
    for t in range(1, T):
        m.addConstr(I[t - 1] <= S[t])  
    for t in range(T):
        m.addConstr(S[t] <= I[t] + meanDemand[t])
    
    
    # optimize
    currTime = time.time()
    m.optimize()
    runTime = time.time() - currTime
    print('running time is %.5f s' % runTime)
    
    # output results
    print('')  
    print('whether ordering at each period :')
    for t in range(T) :
        print('%.2f' % x[t].X, end = ' ')  
    print('')  
    print('cash position at each period :')
    for t in range(T) :
        print('%.2f' % cash[t].getValue(), end = ' ')    
    print('')    
    print('order up to level at each period :')   
    for t in range(T) :
        print('%.2f' % S[t].X, end = ' ') 
    print('')   
    print('end-of-period inventory at each period :')   
    for t in range(T) :
        print('%.2f' % I[t].X, end = ' ')  
    print('') 
    print('ordering quantity at each period :')   
    q = S[0].X - iniInventory
    print('%.2f' % q, end = ' ')  
    for t in range(1, T):
        q = S[t].X - I[t - 1].X
        print('%.2f' % q, end = ' ')  

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))