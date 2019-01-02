# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:16:33 2018

@author: Zhen Chen

@Python version: 3.6

@description: this is for testing a numercial example by adopting Gurobi to solve
              the linear model. 
              The backgroud of this problem is about a small online retailer.
              In the model1, there exits integer variables.
    
"""

from gurobipy import *
import numpy as np
import time


iniInventory = 0 # initial inventory
iniCash = 200 # initial cash
meanDemand = np.ones(150) * 30 # mean demand in each week
T = len(meanDemand) # length of planning horizon
M = 10000

price = np.ones(T) * 50
variCost = np.ones(T) * 20 # unit purchasing cost from the supplier
holdCost = 1
K = np.ones(T) * 100 # overhead costs for each period
alpha = 0 # revenue sharing rate

try :
    # create a Gurobi model
    m = Model("mip_cash_constraint_model1")
    
    
    # decision variables
    Q = {}     # purchasing quantity in each period
    delta = {} # delta_t = 0 means lost sale happens in this period
    RD = {} # realized demand in each perod
    for i in range(T) :
        Q_name = 'Q' + str(i)
        Q[i] = m.addVar(vtype = GRB.CONTINUOUS, name = Q_name)
        delta_name = 'delta_' + str(i)
        delta[i] = m.addVar(vtype = GRB.BINARY, name = delta_name)
        RD_name = 'RD_' + str(i)   
        RD[i] = m.addVar(name = delta_name)
    
    
    # objective function
    I = {}
    I[0] = iniInventory + Q[0] - RD[0]
    for t in range(1, T) :
        I[t] = I[t - 1] + Q[t] - RD[t]
    
    cash = {} # end-of-period cash position for each period
    cash[0] = iniCash + (1 - alpha) * price[0] * RD[0] - variCost[0] * Q[0] - holdCost * I[0] - K[0]
    for t in range(1, T) :
        cash[t] = cash[t - 1] + (1 - alpha) * price[0] * RD[t] - variCost[t] * Q[t] - holdCost * I[t] - K[t]
    
    m.setObjective(cash[T - 1], GRB.MAXIMIZE)
    
    
    # constraints
    m.addConstr(variCost[0] * Q[0] <= iniCash)
    for t in range(1, T) :
        m.addConstr(variCost[t] * Q[t] <= cash[t - 1])   
    
    m.addConstr(RD[0] <= iniInventory + Q[0] + (1 - delta[0]) * M)
    m.addConstr(RD[0] >= iniInventory + Q[0] - (1 - delta[0]) * M)
    m.addConstr(iniInventory + Q[0] <= meanDemand[0] + (1 - delta[0]) * M)
    m.addConstr(RD[0] <= meanDemand[0])
    m.addConstr(RD[0] >= meanDemand[0] - delta[0] * M)
    m.addConstr(iniInventory + Q[0] >= meanDemand[0] + - delta[0] * M)
    for t in range(1, T) :
        m.addConstr(RD[t] <= I[t - 1]+ Q[t] + (1 - delta[t]) * M)
        m.addConstr(RD[t] >= I[t - 1] + Q[t] - (1 - delta[t]) * M)
        m.addConstr(I[t - 1] + Q[t] <= meanDemand[t] + (1 - delta[t]) * M)
        m.addConstr(RD[t] <= meanDemand[t])
        m.addConstr(RD[t] >= meanDemand[t] - delta[t] * M)
        m.addConstr(I[t - 1] + Q[t] >= meanDemand[t] + - delta[t] * M)
    
    
    # optimize
    currTime = time.time()
    m.optimize()
    runTime = time.time() - currTime
    print('running time is %.5f s' % runTime)
    
    # output results
    print('')  
    print('cash position at each period :')
    for t in range(T) :
        print('%.2f' % cash[t].getValue(), end = ' ')    
    print('')    
    print('end-of-period inventory at each period :')   
    for t in range(T) :
        print('%.2f' % I[t].getValue(), end = ' ')  
    print('') 
    print('ordering quantity at each period :')   
    for t in range(T) :
        print('%.2f' % Q[t].X, end = ' ')  

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))