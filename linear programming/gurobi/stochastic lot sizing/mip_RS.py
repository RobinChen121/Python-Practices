# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:00:41 2018

@author: Zhen Chen

@Python version: 3.6

@description: this is a gurobi in python to implement the piesewise method of 
              Roberto Rossi et al. (2015)
    
"""

from gurobipy import *
import numpy as np
from enum import Enum
BoundCriteria = Enum('BoundCriteria', ('lowBound', 'upBound'));

meanDemand = np.array([20, 40, 60, 40])
sigma = meanDemand * 0.25
iniInventory = 0;
fixOrderCost = 100
variCost = 0
holdingCost = 1
penaltyCost = 10

partionNum = 10;
boundCri = BoundCriteria.lowBound
M = 10000;

T = len(meanDemand)

if partionNum == 4 :
    prob = np.array([0.187555, 0.312445, 0.312445, 0.187555])
    means = np.array([-1.43535, -0.415223, 0.415223, 1.43535])
    error = 0.0339052
elif partionNum == 4 :
    prob = np.array([0.04206108420763477, 0.0836356495308449, 0.11074334596058821, 0.1276821455299152, 0.13587777477101692, 0.13587777477101692, 0.1276821455299152, 0.11074334596058821, 0.0836356495308449, 0.04206108420763477])
    means = np.array([-2.133986195498256, -1.3976822972668839, -0.918199946431143, -0.5265753462727588, -0.17199013069262026, 0.17199013069262026, 0.5265753462727588, 0.918199946431143, 1.3976822972668839, 2.133986195498256])
    error = 0.005885974956458359
    

S = {i: fixOrderCost for i in range(0, T)}
h = {i: holdingCost for i in range(0, T)}
v = {i: variCost for i in range(0, T)}
pai = {i: penaltyCost for i in range(0, T)}
I0 = iniInventory;

try:
    # Create a new model
    m = Model("mip_RS")
    
    # Create variables
    x = m.addVars(1, T, GRB.BINARY, name = 'x')
    P = m.addVars(T, T, GRB.BINARY, name = 'P')
    I = m.addVars(1, T, GRB.CONTINUOUS, name = 'I')
    Iplus = m.addVars(1, T, lb = 0, vtype = GRB.CONTINUOUS, name = 'Iplus')
    Iminus = m.addVars(1, T, lb = 0, vtype = GRB.CONTINUOUS, name = 'Iminus')
    
    # Set objective
    m.setObjective(x.prod(S) + Iplus.prod(h) + Iminus.prod(pai) + variCost * (I[0, T - 1, 'C'] - I0), GRB.MINIMIZE)
    
   
    # Add constraints: 
        
    #relationship between x_t and Q_t (I_t + d_t - I_{t-1} <= M*x_t)
    # Q_t >= 0 : I_t + d_t - I_{t-1} >= 0
    for t in range(T) :
        if t > 0 :
            m.addConstr(I[0, t, 'C'] + meanDemand[t] - I[0, t - 1, 'C'] >= 0)
            m.addConstr(I[0, t, 'C'] + meanDemand[t] - I[0, t - 1, 'C'] <= M * x[0, t, 'B'])
        else :
            m.addConstr(I[0, t, 'C'] + meanDemand[t] - I0 >= 0)
            m.addConstr(I[0, t, 'C'] + meanDemand[t] - I0 <= M * x[0, t, 'B'])
     
    # sum_{j=0}^t Pjt == 1           
    for t in range(T) :
        expr = LinExpr()
        for j in range(t) :
            expr += P[t, j, 'B']
        m.addConstr(expr == 1)
    m.addConstrs(P[i, j, 'B'] == 0 for i in range(T) for j in range(T) if j > i)
    
    
    
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))
        
