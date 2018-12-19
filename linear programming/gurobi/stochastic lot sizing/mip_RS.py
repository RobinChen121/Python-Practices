# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:00:41 2018

@author: Zhen Chen

@Python version: 3.6

@description: this is a gurobi in python to implement the piesewise method of 
              Roberto Rossi et al. (2015).
              30 periods 144838 nodes, running time 124s
    
"""

from gurobipy import *
import numpy as np
from enum import Enum
import math
import time


BoundCriteria = Enum('BoundCriteria', ('lowBound', 'upBound'));



meanDemand = np.array([50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50])
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
conSigma = [[0 for x in range(T)] for y in range(T)]
for i in range(T) :
    for j in range(T) :
        sigmaPow = 0
        for k in range(i, j + 1) :
            sigmaPow += math.pow(sigma[k], 2)
        conSigma[i][j] = math.sqrt(sigmaPow)

if partionNum == 4 :
    prob = np.array([0.187555, 0.312445, 0.312445, 0.187555])
    means = np.array([-1.43535, -0.415223, 0.415223, 1.43535])
    error = 0.0339052
elif partionNum == 10 :
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
    x = {}
    P = {}
    I = {}
    Iplus = {}
    Iminus = {}
    for i in range(T) :
        x[i] = m.addVar(vtype = GRB.BINARY)
        P[i] = {}
        for j in range(T) :
            P[i][j] = m.addVar(vtype = GRB.BINARY)
        I[i] = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS)
        Iplus[i] = m.addVar(vtype = GRB.CONTINUOUS)
        Iminus[i] = m.addVar(vtype = GRB.CONTINUOUS)
    
    # Set objective
    totalCosts = LinExpr()
    setupCosts = LinExpr()
    holdCosts = LinExpr()
    penaCosts = LinExpr()
    variCosts = LinExpr()
    for t in range(T) :
        setupCosts += x[t] * S[t]
        holdCosts += Iplus[ t] * h[t]
        penaCosts += Iminus[t] * pai[t]
    variCosts += variCost * (I[T - 1] - I0)
    totalCosts = setupCosts + holdCosts + penaCosts + variCosts
    m.setObjective(totalCosts, GRB.MINIMIZE)
    
   
    # Add constraints: 
        
    #relationship between x_t and Q_t (I_t + d_t - I_{t-1} <= M*x_t)
    # Q_t >= 0 : I_t + d_t - I_{t-1} >= 0
    for t in range(T) :
        if t > 0 :
            m.addConstr(I[t] + meanDemand[t] - I[t - 1] >= 0)
            m.addConstr(I[t] + meanDemand[t] - I[t - 1] <= M * x[t])
        else :
            m.addConstr(I[t] + meanDemand[t] - I0 >= 0)
            m.addConstr(I[t] + meanDemand[t] - I0 <= M * x[t])
     
    # sum_{j=0}^t Pjt == 1 
    for t in range(T) :
        expr = LinExpr()
        for j in range(t + 1) :
            expr += P[j][t]
        m.addConstr(expr == 1)
       
    m.addConstrs(P[j][t] == 0 for i in range(T) for j in range(T) if j > t)
    
    # Pjt >= x_j - sum_{k = j+1}^{t}x_k
    for t in range(T) :       
        for j in range(t + 1) :
            sumxjt = LinExpr()
            for k in range(j + 1, t + 1) :
                sumxjt += x[k]
            m.addLConstr(P[j][t] >= x[j] - sumxjt)
    
    # sum_{j=1}{t}x_j == 0 => P[0][t] == 1, this constraints are important for the extra piecewise constraints          
    for t in range(T) :
        sumxjt2 = LinExpr()
        for j in range(t + 1) :
            sumxjt2 += x[j]
        m.addConstr(x[0] >= 1 - P[0][t])
#    
    # piecewise constraints
    Ipk = LinExpr()
    pmeanPSigma = LinExpr()
    IpkMinuspmeanPSigma = LinExpr()
    for t in range(T) :
        for i in range(partionNum) :  
            PSigma = LinExpr()
            pik = sum(prob[j] for j in range(i + 1))
            Ipk = I[t] * pik
            pmean = 0;
            for k in range(i + 1) :
                pmean += prob[k] * means[k]
            for k in range(t + 1) :
                PSigma += P[k][t] * conSigma[k][t]
            pmeanPSigma = pmean * PSigma
            IpkMinuspmeanPSigma = Ipk - pmeanPSigma
            
            if boundCri == BoundCriteria.lowBound :
                m.addLConstr(Iplus[t] >= IpkMinuspmeanPSigma)
                m.addLConstr(Iminus[t] + I[t] >= IpkMinuspmeanPSigma)
                m.addLConstr(Iminus[t] + I[t] >= 0)
            elif boundCri == BoundCriteria.upBound :
                m.addLConstr(Iplus[t] >= IpkMinuspmeanPSigma + error * PSigma)
                m.addLConstr(Iplus[t] >= error * PSigma)
                m.addLConstr(Iminus[t] + I[t] >= IpkMinuspmeanPSigma + error * PSigma)
                m.addLConstr(Iminus[t] + I[t] >= error * PSigma)
    
    
    m.write('mip_RS_Gurobi.mps')
    currTime = time.time()
    m.optimize()
    runTime = time.time() - currTime
    print('running time is %.5f s' % runTime)
    print('Obj value is : %.2f' % m.objVal)
    if m.status == GRB.Status.OPTIMAL:
        print('x: ', end = '')
        for t in range(T) :
            print('%d' % x[t].X, end = '  ')
        print()
        print('I: ', end = '')
        for t in range(T) :
            print('%d' % I[t].X, end = '  ')
        print()
        print('Iplus: ', end = '')
        for t in range(T) :
            print('%d' % Iplus[t].X, end = '  ')
        print()
        print('Iminus: ', end = '')
        for t in range(T) :
            print('%d' % Iminus[t].X, end = '  ')
        print()
        print('P: ')
        for i in range(T) :        
            for j in range(T) :
                print('%d' % P[i][j].X, end = '  ')
            print()
        print()
    
    
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))
        
