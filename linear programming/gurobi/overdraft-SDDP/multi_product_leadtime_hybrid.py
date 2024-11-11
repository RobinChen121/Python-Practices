#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:13:24 2024

@author: zhenchen

@disp:  hybrid of levering redundance and cut selection
    
    
"""

from gurobipy import *
import itertools
import random
import time
import numpy as np

import sys 
sys.path.append("..") 
from tree import *

    


# for gamma demand
# gamma distribution:mean demand is shape / beta and variance is shape / beta^2
# beta = 1 / scale
# shape = demand * beta
# variance = demand / beta
mean_demands1 =[30, 30, 30, 30, 30] # higher average demand vs lower average demand
mean_demands2 = [i*0.5 for i in mean_demands1] # higher average demand vs lower average demand
# betas = [2, 0.25] # lower variance vs higher variance
# T = len(mean_demands1)

pk1 = [0.25, 0.5, 0.25]
pk2= pk1
xk1 = [mean_demands1[0]-10, mean_demands1[0], mean_demands1[0]+10]
xk2 = [mean_demands2[0]-5, mean_demands2[0], mean_demands2[0]+5]

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

sample_num = 10 # change 1


# for gamma demand
# gamma distribution:mean demand is shape / beta and variance is shape / beta^2
# beta = 1 / scale
# shape = demand * beta
# variance = demand / beta
# mean_demands =[30, 15] # higher average demand vs lower average demand
# betas = [2, 0.25] # lower variance vs higher variance


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

# sample_details1 = [[10, 30], [10, 30], [10, 30]] # change 2
# sample_details2 = [[5, 15], [5, 15], [5, 15]]


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
iter_limit = 100
time_limit = 360
N = 10 # sampled number of scenarios in forward computing, change 3
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

kk = [1, 2, 5, 10]
cut_index = [[]for i in range(iter_limit)]
cut_index_back = [[]for i in range(iter_limit)]
iter = 0
time_pass = 0
start = time.process_time()
# while iter < iter_num:  
while iter < iter_limit: # time_pass < time_limit:   # or
    N = kk[iter] if iter < len(kk) else kk[-1] # this N is the k in the JCAM 2015 paper
    slopes1.append([[[0 for m in range(MM)] for n in range(N)] for t in range(T)])
    slopes2.append([[0 for n in range(N)] for t in range(T)])
    slopes3.append([[[0 for m in range(MM)] for n in range(N)] for t in range(T)])
    intercepts.append([[0 for n in range(N)] for t in range(T-1)])
    q1_values.append([[0 for n in range(N)] for t in range(T)]) 
    qpre1_values.append([[0 for n in range(N)] for t in range(T)]) 
    q2_values.append([[0 for n in range(N)] for t in range(T)]) 
    qpre2_values.append([[0 for n in range(N)] for t in range(T)]) 
    
    cut_index[iter] = [0 for t in range(T-1)]
    cut_index_back[iter] = [0 for t in range(T-1)]
     
    # sample_scenarios1 = generate_scenario_samples_gamma(N, trunQuantile, mean_demands[0], betas[0], T)
    # sample_scenarios2 = generate_scenario_samples_gamma(N, trunQuantile, mean_demands[1], betas[1], T)
    
    sample_scenarios1 = generate_scenarios2(N, sample_num, sample_details1)
    sample_scenarios2 = generate_scenarios2(N, sample_num, sample_details2)
    
    # sample_scenarios1 = generate_scenarios_discrete(N, xk1, pk1, T)
    # sample_scenarios2 = generate_scenarios_discrete(N, xk2, pk2, T)
    
    # sample_scenarios1 = generate_scenarios_normal(N, trunQuantile, mean_demands1, sigmas1)
    # sample_scenarios2 = generate_scenarios_normal(N, trunQuantile, mean_demands2, sigmas2)
     
    # sample_scenarios1 = [[10, 10, 10], [10,10, 30], [10, 30, 10], [10,30, 30],[30,10,10],[30,10,30],[30,30,10],[30,30,30]] # change 4
    # sample_scenarios2 = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [5,15,15],[15,5,5], [15,5, 15], [15,15,5], [15,15,15]]
    
    # forward
    if iter > 0:        
        m.addConstr(theta >= slope_stage1_1[-1][0]*(ini_Is[0]) + slope_stage1_1[-1][1]*(ini_Is[1])\
                            + slope_stage1_2[-1]*(ini_cash-vari_costs[0]*q1-vari_costs[1]*q2-r1*W1+r0*W0-r2*W2)\
                            + slope_stage1_3[-1][0]*q1+slope_stage1_3[-1][1]*q2 + intercept_stage1[-1])        
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    
    # if iter == 14:
    #     m.write('iter' + str(iter+1) + '_main2.lp')    
    #     m.write('iter' + str(iter+1) + '_main2.sol')
    #     pass

    q1_values[iter][0] = [q1.x for n in range(N)]  
    q2_values[iter][0] = [q2.x for n in range(N)]    
    W0_values.append(W0.x)
    W1_values.append(W1.x)
    W2_values.append(W2.x)
    z = m.objVal
    z_values = [[ 0 for t in range(T+1)] for n in range(N)] # for computing the feasible cost
    for n in range(N):
        z_values[n][0] = m.objVal - theta.x
    
    
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q1_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    q2_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    qpre1_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'qpre1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T-1)]
    qpre2_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'qpre2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T-1)]
    I1_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I1_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    I2_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I2_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    cash_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1)+ '^' + str(n+1)) for n in range(N)] for t in range(T)]
    W0_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W1_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W2_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    # B is the quantity of lost sale
    B1_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B1_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    B2_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B2_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
 
    I1_forward_values = [[0 for n in range(N)] for t in range(T)]
    B1_forward_values = [[0 for n in range(N)] for t in range(T)]
    I2_forward_values = [[0 for n in range(N)] for t in range(T)]
    B2_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    W0_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    W1_forward_values = [[0 for n in range(N)] for t in range(T-1)]
    W2_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    
    
    for t in range(T):
        for n in range(N):
            demand1 = sample_scenarios1[n][t]
            demand2 = sample_scenarios2[n][t]
            
            if t == 0:   
                m_forward[t][n].addConstr(I1_forward[t][n] - B1_forward[t][n] == ini_Is[0] - demand1)
                m_forward[t][n].addConstr(I2_forward[t][n] - B2_forward[t][n] == ini_Is[1] - demand2)
                m_forward[t][n].addConstr(cash_forward[t][n] + prices[0]*B1_forward[t][n] + prices[1]*B2_forward[t][n] == ini_cash - overhead_cost[t]\
                                          - vari_costs[0]*q1_values[-1][t][n] -vari_costs[1]*q2_values[-1][t][n] -r1*W1_values[-1] + r0*W0_values[-1]\
                                              -r2*W2_values[-1] + prices[0]*demand1 + prices[1]*demand2)
            else:
                m_forward[t][n].addConstr(I1_forward[t][n] - B1_forward[t][n] == I1_forward_values[t-1][n] + qpre1_values[-1][t-1][n] - demand1)
                m_forward[t][n].addConstr(I2_forward[t][n] - B2_forward[t][n] == I2_forward_values[t-1][n] + qpre2_values[-1][t-1][n] - demand2)
                m_forward[t][n].addConstr(cash_forward[t][n] + prices[0]*B1_forward[t][n] + prices[1]*B2_forward[t][n] == cash_forward_values[t-1][n] - overhead_cost[t]\
                                          - vari_costs[0]*q1_values[-1][t][n] - vari_costs[1]*q2_values[-1][t][n] -r1*W1_forward_values[t-1][n] + r0*W0_forward_values[t-1][n]\
                                              -r2*W2_forward_values[t-1][n] + prices[0]*demand1 + prices[1]*demand2)
             
            if t < T - 1:
                m_forward[t][n].addConstr(qpre1_forward[t][n] == q1_values[-1][t][n]) 
                m_forward[t][n].addConstr(qpre2_forward[t][n] == q2_values[-1][t][n]) 
            if t == T - 1:                   
                m_forward[t][n].setObjective(-prices[0]*(demand1 - B1_forward[t][n])-prices[1]*(demand2 - B2_forward[t][n])\
                                             - unit_salvages[0]*I1_forward[t][n]- unit_salvages[1]*I2_forward[t][n], GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(overhead_cost[t] + vari_costs[0]*q1_forward[t][n] + vari_costs[1]*q2_forward[t][n]\
                                             - prices[0]*(demand1 - B1_forward[t][n]) - prices[1]*(demand2 - B2_forward[t][n])\
                                             + r2*W2_forward[t][n]\
                                             + r1*W1_forward[t][n] - r0*W0_forward[t][n] + theta_forward[t][n], GRB.MINIMIZE)  
                    
                m_forward[t][n].addConstr(W1_forward[t][n] <= U) 
                m_forward[t][n].addConstr(cash_forward[t][n] - vari_costs[0]*q1_forward[t][n] - vari_costs[1]*q2_forward[t][n] - W0_forward[t][n]\
                                          + W1_forward[t][n] + W2_forward[t][n] == overhead_cost[t+1]) 
                m_forward[t][n].addConstr(theta_forward[t][n] >= theta_iniValue*(T-1-t))                  
            
            # put those cuts in the back
            if iter > 0 and t < T - 1:
                for i in range(iter-1):
                    nn = cut_index[i][t] # i is the iter index
                    m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[i][t][nn][0]*(I1_forward[t][n]+ qpre1_forward[t][n])\
                                                  + slopes1[i][t][nn][1]*(I2_forward[t][n]+ qpre2_forward[t][n])\
                                                      + slopes3[i][t][nn][0]*q1_forward[t][n] + slopes3[i][t][nn][1]*q2_forward[t][n]\
                                                  + slopes2[i][t][nn]*(cash_forward[t][n]- vari_costs[0]*q1_forward[t][n]- vari_costs[1]*q2_forward[t][n]-r2*W2_forward[t][n]\
                                             - r1*W1_forward[t][n]+r0*W0_forward[t][n])\
                                                + intercepts[i][t][nn])
            NewJustAdd = True
            loop_no = 0 # necessary
            while NewJustAdd:      
                NewJustAdd = False            
                if iter > 0 and t < T - 1: 
                    nn = cut_index[iter-1][t] # default initiallly select the first cut
                    m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[iter-1][t][nn][0]*(I1_forward[t][n]+ qpre1_forward[t][n])\
                                              + slopes1[iter-1][t][nn][1]*(I2_forward[t][n]+ qpre2_forward[t][n])\
                                                  + slopes3[iter-1][t][nn][0]*q1_forward[t][n] + slopes3[iter-1][t][nn][1]*q2_forward[t][n]\
                                              + slopes2[iter-1][t][nn]*(cash_forward[t][n]- vari_costs[0]*q1_forward[t][n]- vari_costs[1]*q2_forward[t][n]-r2*W2_forward[t][n]\
                                         - r1*W1_forward[t][n]+r0*W0_forward[t][n])\
                                            + intercepts[iter-1][t][nn])
                    
                # optimize
                m_forward[t][n].Params.LogToConsole = 0
                m_forward[t][n].optimize()
                # if iter == 1 and t == 0:
                #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '.lp') 
                #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '.sol') 
                #     pass
            
                I1_forward_values[t][n] = I1_forward[t][n].x 
                I2_forward_values[t][n] = I2_forward[t][n].x 
                               
                B1_forward_values[t][n] = B1_forward[t][n].x  
                B2_forward_values[t][n] = B2_forward[t][n].x 
                cash_forward_values[t][n] = cash_forward[t][n].x 
        
                if t < T - 1: # for computing confidence interval
                    z_values[n][t+1] = m_forward[t][n].objVal - theta_forward[t][n].x
                else:
                    z_values[n][t+1] = m_forward[t][n].objVal
                                        
                if t < T - 1:
                    q1_values[iter][t+1][n] = q1_forward[t][n].x
                    q2_values[iter][t+1][n] = q2_forward[t][n].x
                    qpre1_values[iter][t][n] = qpre1_forward[t][n].x
                    qpre2_values[iter][t][n] = qpre2_forward[t][n].x
                    W1_forward_values[t][n] = W1_forward[t][n].x
                    W0_forward_values[t][n] = W0_forward[t][n].x
                    W2_forward_values[t][n] = W2_forward[t][n].x
                
                if iter > 0 and t < T - 1:                    
                    N2 = kk[iter-1] if iter <= len(kk) else kk[-1]
                    values = [0 for k in range(N2)]     
                    for k in range(N2):
                        values[k] = slopes1[iter-1][t][k][0]*(I1_forward[t][n].x + qpre1_forward[t][n].x)\
                                                  + slopes1[iter-1][t][k][1]*(I2_forward[t][n].x+ qpre2_forward[t][n].x)\
                                                      + slopes3[iter-1][t][k][0]*q1_forward[t][n].x + slopes3[iter-1][t][k][1]*q2_forward[t][n].x\
                                                  + slopes2[iter-1][t][k]*(cash_forward[t][n].x- vari_costs[0]*q1_forward[t][n].x- vari_costs[1]*q2_forward[t][n].x-r2*W2_forward[t][n].x\
                                             - r1*W1_forward[t][n].x+r0*W0_forward[t][n].x)\
                                                + intercepts[iter-1][t][k]
                    if np.argmax(values) != nn:                       
                        NewJustAdd = True
                        loop_no += 1
                        if loop_no > 3:
                            break
                        cut_index[iter - 1][t] = np.argmax(values)
                        m_forward[t][n].remove(m_forward[t][n].getConstrs()[-1])
                
    # backward
    m_backward = [[[Model() for s in range(sample_num)] for n in range(N)] for t in range(T)]
    q1_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'q1_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T - 1)] 
    qpre1_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'qpre1_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T-1)] 
    q2_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'q2_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T - 1)] 
    qpre2_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'qpre2_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T-1)] 
    
    I1_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'I1_' + str(t+1) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T)]
    I2_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'I2_' + str(t+1) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T)]
    
    cash_backward = [[[m_backward[t][n][s].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T)]    
    W0_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T - 1)] 
    W1_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T - 1)] 
    W2_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T - 1)] 
    
    # B is the quantity of lost sale
    B1_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'B1_' + str(t+1) + '^' + str(n+1)) for s in range(sample_num)] for n in range(N)] for t in range(T)]
    B2_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'B2_' + str(t+1) + '^' + str(n+1)) for s in range(sample_num)] for n in range(N)] for t in range(T)]    
    theta_backward = [[[m_backward[t][n][s].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for s in range(sample_num)] for n in range(N)] for t in range(T - 1)]
    
    intercept_values = [[[0 for s in range(sample_num)] for n in range(N)] for t in range(T)]
    slope1_values = [[[[0  for m in range(MM)] for s in range(sample_num)] for n in range(N)] for t in range(T)] 
    slope2_values = [[[0 for s in range(sample_num)] for n in range(N)] for t in range(T)] 
    slope3_values = [[[[0 for m in range(MM)] for s in range(sample_num)] for n in range(N)] for t in range(T)]
    
    for t in range(T-1, -1, -1):  
        demand_temp = [sample_details1[t], sample_details2[t]]
        demand_all = list(itertools.product(*demand_temp))
        # demand_all2 = [[demand_all[s][0], demand_all[s][1]] for s in range(sample_num)]
        for n in range(N):      
            S = sample_num # should revise, should be S^2
            Ip1Ip2W0_values = []
            Ip1Ip2W1_values = []
            Ip1Ip2W2_values = []
            Ip1In2W0_values = []
            Ip1In2W1_values = []
            Ip1In2W2_values = []
            In1Ip2W0_values = []
            In1Ip2W1_values = []
            In1Ip2W2_values = []
            In1In2W0_values = []
            In1In2W1_values = []
            In1In2W2_values = []
            StateChange = False
            for s in range(S):
                demand1 = demand_all[s][0] # sample_details1[t][s]  # 
                demand2 = demand_all[s][1] # sample_details2[t][s]  # 
                # demand1 = sample_details1[t][s]  
                # demand2 = sample_details2[t][s]  
                thisEndCash = 0.0
                if s > 0:
                    if t == 0:
                        thisEndI1 = ini_Is[0] - demand1
                        thisB1 = -min(ini_Is[0] - demand1, 0)
                        thisEndI2 = ini_Is[1] - demand2
                        thisB2 = -min(ini_Is[1] - demand2, 0)
                        if t < T - 1:
                            thisEndCash = ini_cash - overhead_cost[t] - vari_costs[0]*q1_values[-1][t][n]-vari_costs[1]*q2_values[-1][t][n]\
                                                          - r2*W2_values[-1] - r1*W1_values[-1] + r0*W0_values[-1]\
                                                          + prices[0]*(demand1 - thisB1)+ prices[1]*(demand2 - thisB2) - vari_costs[0]*lastq1 - vari_costs[1]*lastq2 - overhead_cost[t+1]
                    else:
                        thisEndI1 = I1_forward_values[t-1][n] + qpre1_values[-1][t-1][n] - demand1
                        thisB1 = -min(I1_forward_values[t-1][n] + qpre1_values[-1][t-1][n] - demand1, 0)
                        thisEndI2 = I2_forward_values[t-1][n] + qpre2_values[-1][t-1][n] - demand2
                        thisB2 = -min(I2_forward_values[t-1][n] + qpre2_values[-1][t-1][n] - demand2, 0)
                        if t < T - 1:
                            thisEndCash = cash_forward_values[t-1][n]- overhead_cost[t]\
                                                          - vari_costs[0]*q1_values[-1][t][n]- vari_costs[1]*q2_values[-1][t][n]\
                                                         - r2*W2_forward_values[t-1][n]- r1*W1_forward_values[t-1][n]\
                                                          + r0*W0_forward_values[t-1][n] + prices[0]*(demand1 - thisB1) - vari_costs[0]*lastq1 - overhead_cost[t+1]\
                                                              + prices[1]*(demand2 - thisB2) - vari_costs[1]*lastq2
                    if thisEndI1 >= 0 and thisEndI2 >= 0 and thisEndCash >= 0:
                        if len(Ip1Ip2W0_values) > 0:
                            pi, rhs = Ip1Ip2W0_values 
                        else:
                            StateChange = True
                    elif thisEndI1 >= 0 and thisEndI2 >= 0 and 0 > thisEndCash >= -U :
                        if len(Ip1Ip2W1_values) > 0:
                            pi, rhs = Ip1Ip2W1_values 
                        else:
                            StateChange = True
                    elif thisEndI1 >= 0 and thisEndI2 >= 0 and thisEndCash < -U :
                        if len(Ip1Ip2W2_values) > 0:
                            pi, rhs = Ip1Ip2W2_values 
                        else:
                            StateChange = True                   
                    elif thisEndI1 >= 0 and thisEndI2 < 0 and thisEndCash >= 0:
                        if len(Ip1In2W0_values) > 0:
                            pi, rhs = Ip1In2W0_values 
                        else:
                            StateChange = True
                    elif thisEndI1 >= 0 and thisEndI2 < 0 and 0 > thisEndCash >= -U:
                        if len(Ip1In2W1_values) > 0:
                            pi, rhs = Ip1In2W1_values 
                        else:
                            StateChange = True
                    elif thisEndI1 >= 0 and thisEndI2 < 0 and thisEndCash < -U:
                        if len(Ip1In2W2_values) > 0:
                            pi, rhs = Ip1In2W2_values 
                        else:
                            StateChange = True
                    elif thisEndI1 < 0 and thisEndI2 >= 0 and thisEndCash >= 0:
                        if len(In1Ip2W0_values) > 0:
                            pi, rhs = In1Ip2W0_values 
                        else:
                            StateChange = True
                    elif thisEndI1 < 0 and thisEndI2 >= 0 and 0 > thisEndCash >= -U:
                        if len(In1Ip2W1_values) > 0:
                            pi, rhs = In1Ip2W1_values 
                        else:
                            StateChange = True
                    elif thisEndI1 < 0 and thisEndI2 >= 0 and thisEndCash < -U:
                        if len(In1Ip2W2_values ) > 0:
                            pi, rhs = In1Ip2W2_values 
                        else:
                            StateChange = True
                    elif thisEndI1 < 0 and thisEndI2 < 0 and thisEndCash > 0:
                        if len(In1In2W0_values ) > 0:
                            pi, rhs = In1In2W0_values 
                        else:
                            StateChange = True
                    elif thisEndI1 < 0 and thisEndI2 < 0 and 0 > thisEndCash > -U:
                        if len(In1In2W1_values ) > 0:
                            pi, rhs = In1In2W1_values 
                        else:
                            StateChange = True
                    elif thisEndI1 < 0 and thisEndI2 < 0 and thisEndCash < -U:
                        if len(In1In2W2_values ) > 0:
                            pi, rhs = In1In2W2_values 
                        else:
                            StateChange = True
                
                if s == 0 or StateChange == True:
                    if t == T - 1:                   
                        m_backward[t][n][s].setObjective(-prices[0]*(demand1 - B1_backward[t][n][s])-prices[1]*(demand2 - B2_backward[t][n][s])\
                                                         - unit_salvages[0]*I1_backward[t][n][s]- unit_salvages[1]*I2_backward[t][n][s], GRB.MINIMIZE)
                    else:
                        m_backward[t][n][s].setObjective(overhead_cost[t] + vari_costs[0]*q1_backward[t][n][s] + vari_costs[1]*q2_backward[t][n][s]\
                                                         - prices[0]*(demand1 - B1_backward[t][n][s])- prices[1]*(demand2 - B2_backward[t][n][s])\
                                                         + r2*W2_backward[t][n][s]
                                                         + r1*W1_backward[t][n][s] - r0*W0_backward[t][n][s] + theta_backward[t][n][s], GRB.MINIMIZE)  
                    if t == 0:   
                        m_backward[t][n][s].addConstr(I1_backward[t][n][s] - B1_backward[t][n][s] == ini_Is[0] - demand1)
                        m_backward[t][n][s].addConstr(I2_backward[t][n][s] - B2_backward[t][n][s] == ini_Is[1] - demand2)
                        m_backward[t][n][s].addConstr(cash_backward[t][n][s] == ini_cash - overhead_cost[t] - vari_costs[0]*q1_values[-1][t][n]-vari_costs[1]*q2_values[-1][t][n]\
                                                      - r2*W2_values[-1] - r1*W1_values[-1] + r0*W0_values[-1]\
                                                      + prices[0]*(demand1 - B1_backward[t][n][s])+ prices[1]*(demand2 - B2_backward[t][n][s]))
                    else:
                        m_backward[t][n][s].addConstr(I1_backward[t][n][s] - B1_backward[t][n][s] == I1_forward_values[t-1][n] + qpre1_values[-1][t-1][n] - demand1)
                        m_backward[t][n][s].addConstr(I2_backward[t][n][s] - B2_backward[t][n][s] == I2_forward_values[t-1][n] + qpre2_values[-1][t-1][n] - demand2)
                        m_backward[t][n][s].addConstr(cash_backward[t][n][s] + prices[0]*B1_backward[t][n][s]+ prices[1]*B2_backward[t][n][s] == cash_forward_values[t-1][n]- overhead_cost[t]\
                                                      - vari_costs[0]*q1_values[-1][t][n]- vari_costs[1]*q2_values[-1][t][n]\
                                                     - r2*W2_forward_values[t-1][n]- r1*W1_forward_values[t-1][n]\
                                                      + r0*W0_forward_values[t-1][n] + prices[0]*demand1+ prices[1]*demand2)
                
                    if t < T - 1:
                        m_backward[t][n][s].addConstr(qpre1_backward[t][n][s] == q1_values[-1][t][n]) 
                        m_backward[t][n][s].addConstr(qpre2_backward[t][n][s] == q2_values[-1][t][n]) 
                    if t < T - 1:                   
                        m_backward[t][n][s].addConstr(W1_backward[t][n][s] <= U)
                        m_backward[t][n][s].addConstr(cash_backward[t][n][s] -vari_costs[0]*q1_backward[t][n][s]- vari_costs[1]*q2_backward[t][n][s] - W0_backward[t][n][s]\
                                                      + W1_backward[t][n][s] + W2_backward[t][n][s] == overhead_cost[t+1])
                        m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= theta_iniValue*(T-1-t))
                        
                    # put those cuts in the back
                    if iter > 0 and t < T - 1:
                        for i in range(iter - 1):
                            nn = cut_index_back[i][t]
                            m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= slopes1[i][t][nn][0]*(I1_backward[t][n][s]+ qpre1_backward[t][n][s])\
                                                              + slopes1[i][t][nn][1]*(I2_backward[t][n][s]+ qpre2_backward[t][n][s])\
                                                          + slopes3[i][t][nn][0]*q1_backward[t][n][s]+ slopes3[i][t][nn][1]*q2_backward[t][n][s]\
                                                          + slopes2[i][t][nn]*(cash_backward[t][n][s]- vari_costs[0]*q1_backward[t][n][s]- vari_costs[1]*q2_backward[t][n][s]\
                                                                               -r2*W2_backward[t][n][s]- r1*W1_backward[t][n][s]+r0*W0_backward[t][n][s])\
                                                        + intercepts[i][t][nn])
                    
                    NewJustAdd = True
                    loop_no = 0
                    while NewJustAdd:
                        NewJustAdd = False
                        if iter > 0 and t < T - 1:
                            nn = cut_index_back[iter - 1][t]
                            m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= slopes1[iter - 1][t][nn][0]*(I1_backward[t][n][s]+ qpre1_backward[t][n][s])\
                                                              + slopes1[iter - 1][t][nn][1]*(I2_backward[t][n][s]+ qpre2_backward[t][n][s])\
                                                          + slopes3[iter - 1][t][nn][0]*q1_backward[t][n][s]+ slopes3[iter - 1][t][nn][1]*q2_backward[t][n][s]\
                                                          + slopes2[iter - 1][t][nn]*(cash_backward[t][n][s]- vari_costs[0]*q1_backward[t][n][s]- vari_costs[1]*q2_backward[t][n][s]\
                                                                               -r2*W2_backward[t][n][s]- r1*W1_backward[t][n][s]+r0*W0_backward[t][n][s])\
                                                        + intercepts[iter - 1][t][nn])
                        # optimize
                        m_backward[t][n][s].Params.LogToConsole = 0
                        m_backward[t][n][s].optimize()                  
                        
                        if iter > 0 and t < T - 1:                    
                            N2 = kk[iter-1] if iter <= len(kk) else kk[-1]
                            values = [0 for k in range(N2)]     
                            for k in range(N2):
                                values[k] = slopes1[iter-1][t][k][0]*(I1_backward[t][n][s].x + qpre1_backward[t][n][s].x)\
                                                          + slopes1[iter-1][t][k][1]*(I2_backward[t][n][s].x + qpre2_backward[t][n][s].x)\
                                                              + slopes3[iter-1][t][k][0]*q1_backward[t][n][s].x + slopes3[iter-1][t][k][1]*q2_backward[t][n][s].x\
                                                          + slopes2[iter-1][t][k]*(cash_backward[t][n][s].x - vari_costs[0]*q1_backward[t][n][s].x- vari_costs[1]*q2_backward[t][n][s].x - r2*W2_backward[t][n][s].x\
                                                     - r1*W1_backward[t][n][s].x + r0*W0_backward[t][n][s].x)\
                                                        + intercepts[iter-1][t][k]
                            if np.argmax(values) != nn:                       
                                NewJustAdd = True
                                loop_no += 1
                                if loop_no > 3:
                                    break
                                cut_index_back[iter - 1][t] = np.argmax(values)
                                m_backward[t][n][s].remove(m_backward[t][n][s].getConstrs()[-1])
                
                        
                        pi = m_backward[t][n][s].getAttr(GRB.Attr.Pi)
                        rhs = m_backward[t][n][s].getAttr(GRB.Attr.RHS)
                        
                        if t < T - 1:
                            lastq1 = q1_backward[t][n][s].X
                            lastq2 = q2_backward[t][n][s].X
                            thisEndCash = cash_backward[t][n][s].X - vari_costs[0]*lastq1  - vari_costs[0]*lastq2 - overhead_cost[t+1] 
                        thisEndI1 = I1_backward[t][n][s].X if I1_backward[t][n][s].X > 0 else  -B1_backward[t][n][s].X
                        thisEndI2 = I2_backward[t][n][s].X if I2_backward[t][n][s].X > 0 else  -B2_backward[t][n][s].X               
                        if thisEndI1 >= 0 and thisEndI2 >= 0 and thisEndCash >= 0:
                            Ip1Ip2W0_values = [pi, rhs]
                        elif thisEndI1 >= 0 and thisEndI2 >= 0  and 0 > thisEndCash >= -U :
                            Ip1Ip2W1_values = [pi, rhs]
                        elif thisEndI1 >= 0 and thisEndI2 >= 0  and thisEndCash < -U :
                            Ip1Ip2W2_values = [pi, rhs]                      
                        elif thisEndI1 >= 0 and thisEndI2 < 0 and thisEndCash >= 0:
                            Ip1In2W0_values = [pi, rhs]
                        elif thisEndI1 >= 0 and thisEndI2 < 0 and 0 > thisEndCash >= -U:
                            Ip1In2W1_values = [pi, rhs]
                        elif thisEndI1 >= 0 and thisEndI2 < 0 and thisEndCash < -U:
                            Ip1In2W2_values = [pi, rhs]
                        elif thisEndI1 < 0 and thisEndI2 >= 0 and thisEndCash >= 0:
                            In1Ip2W0_values = [pi, rhs]
                        elif thisEndI1 < 0 and thisEndI2 >= 0 and 0 > thisEndCash >= -U:
                            In1Ip2W1_values = [pi, rhs]
                        elif thisEndI1 < 0 and thisEndI2 >= 0 and thisEndCash < -U:
                            In1Ip2W2_values = [pi, rhs]
                        elif thisEndI1 < 0 and thisEndI2 < 0 and thisEndCash >= 0:
                            In1In2W0_values = [pi, rhs]
                        elif thisEndI1 < 0 and thisEndI2 < 0 and 0 > thisEndCash >= -U:
                            In1In2W1_values = [pi, rhs]
                        elif thisEndI1 < 0 and thisEndI2 < 0 and thisEndCash < -U:
                            In1In2W2_values = [pi, rhs]
                        StateChange = False
                
                # if iter == 8 and t == 1 and n == 0:
                #     m_backward[t][n][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-' + str(s+1) + '-mback.lp') 
                #     m_backward[t][n][s].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-' + str(s+1) + '-mabck.sol') 
                #     filename = 'iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '-' + str(s+1) + '-m.txt'
                #     with open(filename, 'w') as f:
                #         f.write('demand1=' +str(demand1)+'\n')
                #         f.write('demand2=' +str(demand2)+'\n')
                #         f.write(str(pi))     
                #     pass
            
                num_con = len(pi)
                if t < T - 1:
                    # important
                    intercept_values[t][n][s] += -pi[0]*demand1 -pi[1]*demand2 + pi[2]*prices[0]*demand1+pi[2]*prices[1]*demand2\
                                                 - pi[2]*overhead_cost[t]- prices[0]*demand1 - prices[1]*demand2+ overhead_cost[t+1] # + pi[3]*U + pi[4]*overhead_cost[t+1]-pi[5]*theta_iniValue*(T-1-t) 
                else:
                    intercept_values[t][n][s] += -pi[0]*demand1 -pi[1]*demand2 + pi[2]*prices[0]*demand1+pi[2]*prices[1]*demand2\
                                                 - pi[2]*overhead_cost[t] - prices[0]*demand1 - prices[1]*demand2
                for sk in range(5, num_con):
                    intercept_values[t][n][s] += pi[sk]*rhs[sk]
                
                slope1_values[t][n][s] = [pi[0], pi[1]]                                                         
                slope2_values[t][n][s] = pi[2]
                if t < T -1:
                    slope3_values[t][n][s] = [pi[3], pi[4]]
                
            avg_intercept = sum(intercept_values[t][n]) / S
            avg_slope1 = np.mean(np.array(slope1_values[t][n]), axis=0).tolist()
            avg_slope2 = sum(slope2_values[t][n]) / S
            avg_slope3 = np.mean(np.array(slope3_values[t][n]), axis=0).tolist()
            if t == 0:
                slope_stage1_1.append(avg_slope1)
                slope_stage1_2.append(avg_slope2)
                slope_stage1_3.append(avg_slope3)
                intercept_stage1.append(avg_intercept)
                pass
            else:
                slopes1[-1][t-1][n] = avg_slope1 
                slopes2[-1][t-1][n] = avg_slope2  
                slopes3[-1][t-1][n] = avg_slope3
                intercepts[-1][t-1][n] = avg_intercept  

    iter += 1
    time_pass = time.process_time() - start

end = time.process_time()
print('********************************************')
print('no enhancement')
print('sample numer is %d and scenario number is %d ' % (sample_num, N))
print('planning horizon length is T = %d ' % T)
print('final expected total profits after %d iteration is %.2f' % (iter, -z))
print('ordering Q1 and Q2 in the first peiod is %.2f and %.2f' % (q1_values[iter-1][0][0], q2_values[iter-1][0][0]))
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time) 
z_lb, z_ub, z_mean = compute_ub(z_values) # for computing confidence interval
lb = -np.mean(np.sum(z_values, axis=1))
print('expected lower bound gap is %.2f' % lb)  
gap2 = abs((z+lb)/z)
print('lower bound and upper bound gap is %.2f%%' % (100*gap2))  
print('confidence interval for expected objective is [%.2f,  %.2f]' % (-z_ub, -z_lb))  
