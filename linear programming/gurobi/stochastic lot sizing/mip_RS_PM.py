# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:42:38 2018

@author: Zhen Chen

@Python version: 3.6

@description:  this is a class to implement the PM setting method of Tunc et al. (2018) to solve 
                single-item stochastic lot sizing problem.
                30 periods 0 nodes, running time 1.42s, 50 periods 0 nodes, running time 65s
    
"""

from gurobipy import *
import math
import numpy as np
import time



meanDemand = np.array([50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50])
sigma = meanDemand * 0.25
iniInventory = 0;
fixOrderCost = 100
variCost = 0
holdingCost = 1
penaltyCost = 10

partionNum = 10;
M = 10000;

T = len(meanDemand)
cumDemand = np.cumsum(meanDemand)
conSigma = [[0 for x in range(T)] for y in range(T)] ## initialize a 2-D array
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
    # creat a new model
    model = Model("mip_RS_PM")

    # creat variables
    x = {}
    q = {}
    H = {}
    for i in range(T) :
        x[i] = {}
        q[i] = {}
        H[i] = {}
        for j in range(T) :
            if j < i :
                x_name = 'x' + str(i) + str(j)
                q_name = 'q' + str(i) + str(j)
                x[i][j] = model.addVar(lb = 0, ub = 0, vtype = GRB.BINARY, name = x_name)
                q[i][j] = model.addVar(lb = 0, ub = 0, vtype = GRB.CONTINUOUS, name = q_name)
            else :   
                x_name = 'x' + str(i) + str(j)
                q_name = 'q' + str(i) + str(j)
                q[i][j] = model.addVar(lb = 0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = q_name)
                x[i][j] = model.addVar(vtype = GRB.BINARY, name = x_name)
            H[i][j] = {}
            for t in range(T):
                H_name = 'H' + str(i) + str(j) + str(t)
                if t >= i and t <= j :
                    H[i][j][t] = model.addVar(lb = 0, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS, name = H_name)
                else :
                    H[i][j][t] = model.addVar(lb = 0, ub = 0, vtype = GRB.CONTINUOUS, name = H_name)
    
    # set objective
    totalCosts = LinExpr()
    setupCosts = LinExpr()
    holdCosts = LinExpr()
    penaCosts = LinExpr()
    Dxij = LinExpr()
    
    for i in range(T) :
        for j in range(T) :
            setupCosts += x[i][j] * S[i] 
            for t in range(i, j + 1) :
                Dxij += -h[i] * cumDemand[t] * x[i][j]
                holdCosts += q[i][j] * h[i]
                penaCosts += H[i][j][t] * (h[i] + pai[i])
    totalCosts = setupCosts + holdCosts + penaCosts + Dxij
    model.setObjective(totalCosts, GRB.MINIMIZE)
    
    
    # add constraints
    
	# sum_{i=0}^T x_{1, i} = 1
	# sum_{i=0}^T x_{i, T} = 1;
	# sum_{i=0}^t x_{i, t} = sum_{j=t+1}^T x_{t, j}
    expr_x1i = LinExpr()
    expr_xiT = LinExpr()
    for i in range(T) :
        expr_x1i += x[0][i]
        expr_xiT += x[i][T - 1]
    model.addConstr(expr_x1i == 1)
    model.addConstr(expr_xiT == 1)
    for t in range(T - 1) :
        expr_xit = LinExpr()
        expr_xtj = LinExpr()
        for i in range(t + 1) :
            expr_xit += x[i][t]
        for j in range(t + 1, T) :
            expr_xtj += x[t + 1][j]
        model.addConstr(expr_xit == expr_xtj)
    
    # q_{i,j} <= Mx_{i,j}
    for i in range(T) :
        for j in range(T) :
            model.addConstr(q[i][j] <= M * x[i][j])
           
    # sum_{i=0}^t q_{i, t} <= sum_{j=t+1}^T q_{t, j}
    for t in range(T - 1) :
        expr_qit = LinExpr()
        expr_qtj = LinExpr()
        for i in range(t + 1) :
            expr_qit += q[i][t]
        for j in range(t + 1, T) :
            expr_qtj += q[t + 1][j]
        model.addConstr(expr_qit <= expr_qtj)
        
    # piecewise constraints
    I = LinExpr()
    for i in range(T) :
        for j in range(i, T) :
            for t in range(i, j + 1) :
                I = q[i][j] - x[i][j] * cumDemand[t]
                for k in range(partionNum) :
                    pik = prob[range(k + 1)].sum() # slope
                    pmean = prob[range(k + 1)].dot(means[range(k + 1)]) # intercept
                    model.addConstr(H[i][j][t] + I >= pik * I - x[i][j] * pmean * conSigma[i][t])
    
    # model.write('mip_RS_PM_Gurobi.lp')
    currTime = time.time()
    model.optimize()
    runTime = time.time() - currTime
    print('running time is %.5f s' % runTime)
    
#    print('setup costs is: %f' % setupCosts.getValue())
#    print('penalty costs is: %f' % penaCosts.getValue())
#    mark_holdCosts = holdCosts.getValue() + Dxij.getValue()
#    print('hold costs is: %f' % mark_holdCosts)
    if model.status == GRB.Status.OPTIMAL : 
#        print('x: ', end = '')
#        print('\n')
#        for i in range(T) :
#            for j in range(T) :
#                print('%d' % x[i][j].X, end = '  ')
#            print('\n')
#        print('*************************')
#        print('q: ', end = '')
#        print('\n')
#        for i in range(T) :
#            for j in range(T) :
#                print('%d' % q[i][j].X, end = '  ')
#            print('\n')
#        print('H: ', end = '')
#        print('\n')
#        for i in range(T) :
#            for j in range(T) :
#                for t in range(T) :
#                    print('%d' % H[i][j][t].X, end = '  ')
#                print('\n')
#            print('\n')
        
        z = [0 for i in range(T)]
        quantity = [0 for i in range(T)]
        I = [0 for i in range(T)]
        lastQ = 0
        for i in range(T) :
            for j in range(T) :
                if x[i][j].X == 1 :
                    z[i] = 1
                    if i == 0 :
                        quantity[i] = q[i][j].X
                        lastQ = quantity[i]
                    else :
                        quantity[i] = q[i][j].X - lastQ
                        lastQ = quantity[i]
        I[0] = quantity[0] + iniInventory - meanDemand[0]
        for i in range(1, T) :
            I[i] = quantity[i] + I[i - 1] - meanDemand[i]
        print('*************************')
        print('z = ')
        print(z)
        print('Q = ')
        print(quantity)
        print('I = ')
        print(I)
    elif model.status == GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % model.status)
        # do IIS, find infeasible constraints
        model.computeIIS()
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
            
    
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))




