#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:51:37 2024

@author: zhenchen

@disp:  transfer the multi stage model to a two stage model.
    
    
"""

from gurobipy import *
import itertools
import random
import time
import numpy as np

import sys 
import os
parent_directory = os.path.abspath('..')
sys.path.append(parent_directory) 
# sys.path.append("..") # for mac
from tree import *



demands = [[30,30,30,30,30],
[50, 46, 38, 28, 14],
[14,23,33,46,50],
[47,30,6,30,54],
[9,30,44,30,8],
[63,27,10,24,1],
[25, 46, 140, 80, 147],
[14,24,71,118,49],
[13,35,79,43,44],
[15,56,19,84,136]]

demand_pattern = 8
mean_demands1 = demands[demand_pattern - 1] # higher average demand vs lower average demand
mean_demands2 = [i*0.5 for i in mean_demands1] # higher average demand vs lower average demand

# pk1 = [0.25, 0.5, 0.25]
# pk2= pk1
# xk1 = [mean_demands1[0]-10, mean_demands1[0], mean_demands1[0]+10]
# xk2 = [mean_demands2[0]-5, mean_demands2[0], mean_demands2[0]+5]

cov1 = 0.25 # lower variance vs higher variance
cov2 = 0.5
sigmas1 = [cov1*i for i in mean_demands1]
sigmas2 = [cov2*i for i in mean_demands2]
T = len(mean_demands1)

ini_Is = [0, 0]
ini_cash = 0
vari_costs = [1, 2]
prices = [5, 10] # lower margin vs higher margin
MM = len(prices)
unit_salvages = [0.5* vari_costs[m] for m in range(MM)]
overhead_cost = [100 for t in range(T)]

r0 = 0  # when it is 0.01, can largely slow the compuational speed
r1 = 0.1
r2 = 2 # penalty interest rate for overdraft exceeding the limit, does not affect computation time
U = 500 # overdraft limit

sample_num = 2 # change 1

# detailed samples in each period
trunQuantile = 0.9999 # affective to the final ordering quantity
sample_details1 = [[0 for i in range(sample_num)] for t in range(T)]
sample_details2 = [[0 for i in range(sample_num)] for t in range(T)]
for t in range(T):
    # sample_details1[t] = generate_samples_gamma(sample_num, trunQuantile, mean_demands1[t], betas[0])
    # sample_details2[t] = generate_samples_gamma(sample_num, trunQuantile, mean_demands2[t], betas[1])
    # sample_details1[t] = generate_samples(sample_num, trunQuantile, mean_demands1[t])
    # sample_details2[t] = generate_samples(sample_num, trunQuantile, mean_demands2[t])
    sample_details1[t] = generate_samples_normal(sample_num, trunQuantile, mean_demands1[t], sigmas1[t])
    sample_details2[t] = generate_samples_normal(sample_num, trunQuantile, mean_demands2[t], sigmas2[t])
    # sample_details1[t] = generate_samples_discrete(sample_num, xk1, pk1)
    # sample_details2[t] = generate_samples_discrete(sample_num, xk2, pk2)

sample_details1 = [[10, 30], [10, 30], [10, 30]] # change 2
sample_details2 = [[5, 15], [5, 15], [5, 15]]

theta_iniValue = -1000 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
q2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_2')
W0 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^0')
W1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^1')
W2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^2')
theta = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_2')

m.setObjective(overhead_cost[0] + vari_costs[0]*q1 + vari_costs[1]*q2 + r2*W2 + r1*W1 - r0*W0 + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))
m.addConstr(W1 <= U)
m.addConstr(-vari_costs[0]*q1 - vari_costs[1]*q2- W0 + W1 + W2 == overhead_cost[0] - ini_cash)


# cuts recording arrays
iter_limit = 500
time_limit = 3600
N = 8 # sampled number of scenarios in forward computing, change 3
slope_stage1_1 = []
slope_stage1_2 = []
slope_stage1_3 = []
intercept_stage1 = []
slopes1 = []
slopes2 = []
slopes3 = []
intercepts = []
q1_values = [] 
qpre1_values = [] 
q2_values = [] 
qpre2_values = [] 
W0_values = []
W1_values = []
W2_values = []

iter = 0
time_pass = 0
stop_condition = 'iter_limit'
start = time.process_time()

while iter < iter_limit and time_pass < time_limit: # and means satifying either one will exist the loop
    
    sample_scenarios1 = generate_scenarios_normal(N, trunQuantile, mean_demands1, sigmas1)
    sample_scenarios2 = generate_scenarios_normal(N, trunQuantile, mean_demands2, sigmas2)
    
    sample_scenarios1 = [[10, 10, 10], [10,10, 30], [10, 30, 10], [10,30, 30],[30,10,10],[30,10,30],[30,30,10],[30,30,30]] # change 4
    sample_scenarios2 = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [5,15,15],[15,5,5], [15,5, 15], [15,15,5], [15,15,15]]
    
    if iter > 0:        
        m.addConstr(theta >= slope[0]*q1 + slope[1]*q2 + slope[2]*W0 + slope[3]*W1 + slope[4]*W2 + intercept)        
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    
    W0_values.append(W0.x)
    W1_values.append(W1.x)
    W2_values.append(W2.x)
    z = m.objVal
    z_values = [[ 0 for t in range(T+1)] for n in range(N)] # for computing the feasible cost
    for n in range(N):
        z_values[n][0] = m.objVal - theta.x
    
    
    m_forward = [Model() for n in range(N)]
    q1_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'q1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    q2_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'q2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    qpre1_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'qpre1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T-1)]
    qpre2_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'qpre2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T-1)]
    I1_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'I1_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    I2_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'I2_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    cash_forward = [[m_forward[n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1)+ '^' + str(n+1)) for n in range(N)] for t in range(T)]
    W0_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W1_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W2_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    # B is the quantity of lost sale
    B1_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'B1_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    B2_forward = [[m_forward[n].addVar(vtype = GRB.CONTINUOUS, name = 'B2_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
 
    I1_forward_values = [[0 for n in range(N)] for t in range(T)]
    B1_forward_values = [[0 for n in range(N)] for t in range(T)]
    I2_forward_values = [[0 for n in range(N)] for t in range(T)]
    B2_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    W0_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    W1_forward_values = [[0 for n in range(N)] for t in range(T-1)]
    W2_forward_values = [[0 for n in range(N)] for t in range(T-1)]
    
    intercepts = [0 for n in range(N)]
    slopes = [[0 for i in range(5)] for n in range(N)]
    for n in range(N):
        demand1 = [0 for t in range(T)]
        demand2 = [0 for t in range(T)]
        for t in range(T):
            demand1[t] = sample_scenarios1[n][t]
            demands[t] = sample_scenarios2[n][t]
        
        neg_revenue_total = LinExpr()      
        for t in range(T):
            neg_revenue_total += -prices[0]*(demand1[t] - B1_forward[t][n]) - prices[1]*(demand2[t] - B2_forward[t][n])
        overhead_total = sum(overhead_cost[1:])
        interest_total = LinExpr()
        for t in range(T-1):
            interest_total += r2*W2_forward[t][n] + r1*W1_forward[t][n] - r0*W0_forward[t][n]
        variCosts_total = LinExpr()
        for t in range(T-1):
            variCosts_total += vari_costs[0]*q1_forward[t][n] + vari_costs[1]*q2_forward[t][n]
        m_forward[n].setObjective(neg_revenue_total+overhead_total+interest_total+variCosts_total, GRB.MINIMIZE)
        
            
        for t in range(T):
            if t == 0:
                m_forward[n].addConstr(I1_forward[t][n] - B1_forward[t][n] == ini_Is[0] - demand1[0])
                m_forward[n].addConstr(I2_forward[t][n] - B2_forward[t][n] == ini_Is[1] - demand2[0])
            else:
                if t == 1:
                    m_forward[n].addConstr(I1_forward[t][n] - B1_forward[t][n] == I1_forward[t-1][n] + q1.x - demand1[t])
                    m_forward[n].addConstr(I2_forward[t][n] - B2_forward[t][n] == I2_forward[t-1][n] + q2.x - demand2[t])
                else:
                    m_forward[n].addConstr(I1_forward[t][n] - B1_forward[t][n] == I1_forward[t-1][n] + q1_forward[t-2][n]- demand1[t])
                    m_forward[n].addConstr(I2_forward[t][n] - B2_forward[t][n] == I2_forward[t-1][n] + q2_forward[t-2][n]- demand2[t])
        
        for t in range(T):
            if t == 0:   
                m_forward[n].addConstr(cash_forward[t][n] + prices[0]*B1_forward[t][n] + prices[1]*B2_forward[t][n] == ini_cash - overhead_cost[t]\
                                          - vari_costs[0]*q1.x -vari_costs[1]*q2.x -r1*W1.x + r0*W0.x\
                                              -r2*W2.x + prices[0]*demand1[t] + prices[1]*demand2[t])
            else:
                m_forward[n].addConstr(cash_forward[t][n] + prices[0]*B1_forward[t][n] + prices[1]*B2_forward[t][n] == cash_forward[t-1][n] - overhead_cost[t]\
                                          - vari_costs[0]*q1_forward[t-1][n] - vari_costs[1]*q2_forward[t-1][n] -r1*W1_forward[t-1][n] + r0*W0_forward[t-1][n]\
                                              -r2*W2_forward[t-1][n] + prices[0]*demand1[t] + prices[1]*demand2[t])
        for t in range(T-1):
            m_forward[n].addConstr(W1_forward[t][n] <= U) 
            m_forward[n].addConstr(cash_forward[t][n] - vari_costs[0]*q1_forward[t][n] - vari_costs[1]*q2_forward[t][n] - W0_forward[t][n]\
                                  + W1_forward[t][n] + W2_forward[t][n] == overhead_cost[t+1])         
        
        # optimize
        m_forward[n].Params.LogToConsole = 0
        m_forward[n].optimize()
        
        pi = m_forward[n].getAttr(GRB.Attr.Pi)
        rhs = m_forward[n].getAttr(GRB.Attr.RHS)
        
        num_con = len(pi)
        for i in range(num_con):
            if i not in [2, 3, 2*T]: 
                intercepts[n] += pi[i]*rhs[i]
        intercepts[n] += -pi[2]*demand1[1] - pi[3]*demand2[1] + pi[2*T]*(ini_cash - overhead_cost[t] + prices[0]*demand1[0] + prices[1]*demand2[0])
        slopes[n] = [pi[1]- vari_costs[0]*pi[2*T], pi[2]- vari_costs[1]*pi[2*T], r0*pi[2*T], -r1*pi[2*T], -r2*pi[2*T]]
    
        intercept = np.mean(intercepts)
        slope = np.mean(slopes, axis = 1)
    iter += 1
    time_pass = time.process_time() - start

end = time.process_time()
print('********************************************')
print('warm start')
print('sample numer is %d and scenario number is %d ' % (sample_num, N))
print('planning horizon length is T = %d ' % T)
print('final expected total profits after %d iteration is %.2f' % (iter, -z))
print('ordering Q1 and Q2 in the first peiod is %.2f and %.2f' % (q1.x, q2.x))
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time) 