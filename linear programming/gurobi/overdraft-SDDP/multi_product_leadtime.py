#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:18:39 2024

@author: zhenchen

@disp:  business overdraft for lead time in 2 product problem;

using random gamma generator seems not very well
"""

from gurobipy import *
import itertools
import random
import time
import numpy as np

import sys 
sys.path.append("..") 
from tree import generate_gamma_sample



def generate_samples(sample_details, N):
    sample_num_1stage = len(sample_details[0][0])
    T = len(sample_details[0])
    MM = len(sample_details)    
    random_samples = [[[0 for t in range(T)] for m in range(MM)] for n in range(N)]      
    for n in range(N):
        random_index1 = np.random.randint(low = 0, high = sample_num_1stage, size = T).tolist()
        random_index2 = np.random.randint(low = 0, high = sample_num_1stage, size = T).tolist()
        for t in range(T):
            index1 = random_index1[t]
            index2 = random_index2[t]
            random_samples[n][0][t] = sample_details[0][t][index1]
            random_samples[n][1][t] = sample_details[1][t][index2]
    return random_samples
    


T = 3
ini_Is = [0, 0]
ini_cash = 0
vari_costs = [1, 2]
prices = [5, 10] # lower margin vs higher margin
MM = len(prices)
unit_salvages = [0.5* vari_costs[m] for m in range(MM)]
overhead_cost = [100 for t in range(T)]

r0 = 0
r1 = 0
r2 = 0 # penalty interest rate for overdraft exceeding the limit
U = 2000 # overdraft limit

sample_num = 2 # sample number in one stage when forming the scenario tree #
scenario_numTotal = sample_num ** T

# gamma distribution:mean demand is shape / beta and variance is shape / beta^2
# beta = 1 / scale
# shape = demand * beta
# variance = demand / beta
mean_demands =[20, 10] # higher average demand vs lower average demand
betas = [10, 1] # lower variance vs higher variance

# detailed samples in each period
trunQuantile = 0.9999 # affective to the final ordering quantity
sample_details = [[[0 for i in range(sample_num)] for t in range(T)] for m in range(MM)]
for m in range(MM):
    for t in range(T):
            sample_details[m][t] = generate_gamma_sample(sample_num, trunQuantile, mean_demands[m], betas[m])

sample_details = [[[10, 20], [10, 20], [10, 20]], [[5, 15], [5, 15], [5, 15]]] #

theta_iniValue = -5000 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
q2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_2')
W0 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^0')
W1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^1')
W2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^2')
theta = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_2')

m.setObjective(vari_costs[0]*q1 + vari_costs[1]*q2 + r2*W2 + r1*W1 - r0*W0 + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))
m.addConstr(W1 <= U)
m.addConstr(-vari_costs[0]*q1 - vari_costs[1]*q2- W0 + W1 + W2 == overhead_cost[0] - ini_cash)

# cuts recording arrays
iter_num = 4
N = 2 # sampled number of scenarios in forward computing 2
slope_stage1_1 = []
slope_stage1_2 = []
slope_stage1_3 = []
intercept_stage1 = []
slopes1 = [[[[0 for m in range(MM)] for n in range(N)] for t in range(T)] for i in range(iter_num)]
slopes2 = [[[0 for n in range(N)] for t in range(T)] for i in range(iter_num)]
slopes3 = [[[[0 for m in range(MM)] for n in range(N)] for t in range(T)] for i in range(iter_num)]
intercepts = [[[0 for n in range(N)] for t in range(T-1)] for i in range(iter_num)]
q1_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
qpre1_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
q2_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
qpre2_values = [[[0 for n in range(N)] for t in range(T)] for iter in range(iter_num)] 
W0_values = [0 for iter in range(iter_num)]
W1_values = [0 for iter in range(iter_num)]
W2_values = [0 for iter in range(iter_num)]

start = time.process_time()
iter = 0
while iter < iter_num:  
    sample_scenarios = generate_samples(sample_details, N)
    sample_scenarios = [[[10, 10, 10], [10, 10, 30], [10, 30, 10], [10,30,30], [30,10,10], [30,10,30],[30,30,10],  [30,30,30]],\
                        [[5, 5, 5], [5, 5, 15], [5, 15, 5], [5,15,15],[15,5,5], [15,5, 15], [15,15,5], [15,15,15]]]
    
    # forward
    if iter > 0:        
        m.addConstr(theta >= slope_stage1_1[-1][0]*(ini_Is[0]) + slope_stage1_1[-1][1]*(ini_Is[1])\
                            + slope_stage1_2[-1]*(ini_cash-vari_costs[0]*q1-vari_costs[1]*q2)\
                            + slope_stage1_3[-1][0]*q1+slope_stage1_3[-1][1]*q2 + intercept_stage1[-1])        
    m.update()
    m.optimize()
    
    if iter == 0:
        m.write('iter' + str(iter+1) + '_main.lp')    
        m.write('iter' + str(iter+1) + '_main.sol')
        pass

    q1_values[iter][0] = [q1.x for n in range(N)]  
    q2_values[iter][0] = [q2.x for n in range(N)] 
    
    W0_values[iter] = W0.x
    W1_values[iter] = W1.x
    W2_values[iter] = W2.x
    z = m.objVal
    
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
            demand1 = sample_scenarios[n][0][t]
            demand2 = sample_scenarios[n][1][t]
            
            if t == 0:   
                m_forward[t][n].addConstr(I1_forward[t][n] - B1_forward[t][n] == ini_Is[0] - demand1)
                m_forward[t][n].addConstr(I2_forward[t][n] - B2_forward[t][n] == ini_Is[1] - demand2)
                m_forward[t][n].addConstr(cash_forward[t][n] + prices[0]*B1_forward[t][n] + + prices[1]*B2_forward[t][n] == ini_cash - overhead_cost[t]\
                                          - vari_costs[0]*q1_values[iter][t][n] -vari_costs[1]*q2_values[iter][t][n] -r1*W1_values[iter] + r0*W0_values[iter]\
                                              -r2*W2_values[iter] + prices[0]*demand1 + prices[1]*demand2)
            else:
                m_forward[t][n].addConstr(I1_forward[t][n] - B1_forward[t][n] == I1_forward_values[t-1][n] + qpre1_values[iter][t-1][n] - demand1)
                m_forward[t][n].addConstr(I2_forward[t][n] - B2_forward[t][n] == I2_forward_values[t-1][n] + qpre2_values[iter][t-1][n] - demand2)
                m_forward[t][n].addConstr(cash_forward[t][n] + prices[0]*B1_forward[t][n] + + prices[1]*B2_forward[t][n] == cash_forward_values[t-1][n] - overhead_cost[t]\
                                          - vari_costs[0]*q1_values[iter][t][n] - vari_costs[1]*q2_values[iter][t][n]-r2*W2_values[iter] -r1*W1_values[iter] + r0*W0_values[iter]\
                                              -r2*W2_values[iter] + prices[0]*demand1 + prices[1]*demand2)
             
            if t < T - 1:
                m_forward[t][n].addConstr(qpre1_forward[t][n] == q1_values[iter][t][n]) 
                m_forward[t][n].addConstr(qpre2_forward[t][n] == q2_values[iter][t][n]) 
            m_forward[t][n].addConstr(B1_forward[t][n] <= demand1) 
            m_forward[t][n].addConstr(B2_forward[t][n] <= demand2) 
            if t == T - 1:                   
                m_forward[t][n].setObjective(-prices[0]*(demand1 - B1_forward[t][n])-prices[1]*(demand2 - B2_forward[t][n])\
                                             - unit_salvages[0]*I1_forward[t][n]- unit_salvages[1]*I2_forward[t][n], GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(overhead_cost[t] + vari_costs[0]*q1_forward[t][n] + vari_costs[1]*q2_forward[t][n]\
                                             - prices[0]*(demand1 - B1_forward[t][n]) - prices[1]*(demand2 - B2_forward[t][n])\
                                             + r2*W2_forward[t][n]\
                                             + r1*W1_forward[t][n] - r0*W0_forward[t][n] + theta_forward[t][n], GRB.MINIMIZE)  
                    
                m_forward[t][n].addConstr(W1_forward[t][n] <= U) 
                m_forward[t][n].addConstr(- vari_costs[0]*q1_forward[t][n] - vari_costs[1]*q2_forward[t][n] - W0_forward[t][n]\
                                          + W1_forward[t][n] + W2_forward[t][n] == overhead_cost[t] - cash_forward_values[t-1][n]) 
                m_forward[t][n].addConstr(theta_forward[t][n] >= theta_iniValue*(T-1-t))                  
            
            # put those cuts in the back
            if iter > 0 and t < T - 1:
                for i in range(iter):
                    for nn in range(N): # N
                        m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[i][t][nn][0]*(I1_forward[t][n]+ qpre1_forward[t][n])\
                                                  + slopes1[i][t][nn][1]*(I2_forward[t][n]+ qpre2_forward[t][n])\
                                                      + slopes3[i][t][nn][0]*q1_forward[t][n] + slopes3[i][t][nn][1]*q2_forward[t][n]\
                                                  + slopes2[i][t][nn]*(cash_forward[t][n]- vari_costs[0]*q1_forward[t][n]- vari_costs[1]*q2_forward[t][n]-r2*W2_forward[t][n]\
                                             - r1*W1_forward[t][n]+r0*W0_forward[t][n])\
                                                + intercepts[i][t][nn])
            
            # optimize
            m_forward[t][n].optimize()
            if iter == 1 and t == 0:
                m_forward[t][n].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '.lp') 
                m_forward[t][n].write('iter' + str(iter+1) + '_sub_' + str(t+1) + '^' + str(n+1) + '.sol') 
                pass
        
            I1_forward_values[t][n] = I1_forward[t][n].x 
            I2_forward_values[t][n] = I2_forward[t][n].x 
                           
            B1_forward_values[t][n] = B1_forward[t][n].x  
            B2_forward_values[t][n] = B2_forward[t][n].x 
            cash_forward_values[t][n] = cash_forward[t][n].x 
        
            if t < T - 1:
                q1_values[iter][t+1][n] = q1_forward[t][n].x
                qpre1_values[iter][t][n] = qpre1_forward[t][n].x
                q2_values[iter][t+1][n] = q2_forward[t][n].x
                qpre2_values[iter][t][n] = qpre2_forward[t][n].x
                W1_forward_values[t][n] = W1_forward[t][n].x
                W0_forward_values[t][n] = W0_forward[t][n].x
                W2_forward_values[t][n] = W2_forward[t][n].x
                
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
        for n in range(N):      
            S = sample_num
            for s in range(S):
                demand1 = sample_scenarios[n][0][t]
                demand2 = sample_scenarios[n][1][t]
                
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
                    m_backward[t][n][s].addConstr(cash_backward[t][n][s] == ini_cash - overhead_cost[t] - vari_costs[0]*q1_values[iter][t][n]-vari_costs[1]*q2_values[iter][t][n]\
                                                  - r2*W2_values[iter]- r2*W2_values[iter] - r1*W1_values[iter] + r0*W0_values[iter]\
                                                  + prices[0]*(demand1 - B1_backward[t][n][s])+ prices[1]*(demand2 - B2_backward[t][n][s]))
                else:
                    m_backward[t][n][s].addConstr(I1_backward[t][n][s] - B1_backward[t][n][s] == I1_forward_values[t-1][n] + qpre1_values[iter][t-1][n] - demand1)
                    m_backward[t][n][s].addConstr(I2_backward[t][n][s] - B2_backward[t][n][s] == I2_forward_values[t-1][n] + qpre2_values[iter][t-1][n] - demand2)
                    m_backward[t][n][s].addConstr(cash_backward[t][n][s] + prices[0]*B1_backward[t][n][s]+ prices[1]*B2_backward[t][n][s] == cash_forward_values[t-1][n]- overhead_cost[t]\
                                                  - vari_costs[0]*q1_values[iter][t][n]- vari_costs[1]*q2_values[iter][t][n]\
                                                 - r2*W2_forward_values[t-1][n]- r1*W1_forward_values[t-1][n]\
                                                  + r0*W0_forward_values[t-1][n] + prices[0]*demand1+ prices[1]*demand2)
            
                if t < T - 1:
                    m_backward[t][n][s].addConstr(qpre1_backward[t][n][s] == q1_values[iter][t][n]) 
                    m_backward[t][n][s].addConstr(qpre2_backward[t][n][s] == q2_values[iter][t][n]) 
                m_backward[t][n][s].addConstr(B1_backward[t][n][s] <= demand1)
                m_backward[t][n][s].addConstr(B2_backward[t][n][s] <= demand2)
                if t < T - 1:                   
                    m_backward[t][n][s].addConstr(W1_backward[t][n][s] <= U)
                    m_backward[t][n][s].addConstr(-vari_costs[0]*q1_backward[t][n][s]- vari_costs[1]*q2_backward[t][n][s] - W0_backward[t][n][s]\
                                                  + W1_backward[t][n][s] + W2_backward[t][n][s] == overhead_cost[t] - cash_forward_values[t-1][n])
                    m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= theta_iniValue*(T-1-t))
                    
                # put those cuts in the back
                if iter > 0 and t < T - 1:
                    for i in range(iter):
                        for nn in range(N): # N
                            m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= slopes1[i][t][nn][0]*(I1_backward[t][n][s]+ qpre1_backward[t][n][s])\
                                                          + slopes1[i][t][nn][1]*(I2_backward[t][n][s]+ qpre2_backward[t][n][s])\
                                                      + slopes3[i][t][nn][0]*q1_backward[t][n][s]+ slopes3[i][t][nn][1]*q2_backward[t][n][s]\
                                                      + slopes2[i][t][nn]*(cash_backward[t][n][s]- vari_costs[0]*q1_backward[t][n][s]- vari_costs[1]*q2_backward[t][n][s]\
                                                                           -r2*W2_backward[t][n][s]- r1*W1_backward[t][n][s]+r0*W0_backward[t][n][s])\
                                                    + intercepts[i][t][nn])
                               
                # optimize
                m_backward[t][n][s].optimize()
                
                pi = m_backward[t][n][s].getAttr(GRB.Attr.Pi)
                rhs = m_backward[t][n][s].getAttr(GRB.Attr.RHS)
                
                num_con = len(pi)
                if t < T - 1:
                    # important
                    intercept_values[t][n][s] += -pi[0]*demand1 -pi[1]*demand2 + pi[2]*prices[0]*demand1+pi[3]*prices[1]*demand2\
                                                 - pi[2]*overhead_cost[t]- pi[3]*overhead_cost[t] - prices[0]*demand1 - prices[1]*demand2+ overhead_cost[t+1] # + pi[3]*U + pi[4]*overhead_cost[t+1]-pi[5]*theta_iniValue*(T-1-t) 
                else:
                    intercept_values[t][n][s] += -pi[0]*demand1 -pi[1]*demand2 + pi[2]*prices[0]*demand1+pi[3]*prices[1]*demand2\
                                                 - pi[2]*overhead_cost[t]- pi[3]*overhead_cost[t] - prices[0]*demand1 - prices[1]*demand2
                for sk in range(4, num_con):
                    intercept_values[t][n][s] += pi[sk]*rhs[sk]
                
                slope1_values[t][n][s] = [pi[0], pi[1]]                                                         
                slope3_values[t][n][s] = [pi[3], pi[4]]
                if t < T -1:
                    slope2_values[t][n][s] = pi[2]
                
            avg_intercept = sum(intercept_values[t][n]) / S
            avg_slope1 = np.mean(np.array(slope1_values[t][n]), axis=0).tolist()
            avg_slope2 = sum(slope2_values[t][n]) / S
            avg_slope3 = np.mean(np.array(slope3_values[t][n]), axis=0).tolist()
            if t == 0:
                slope_stage1_1.append(avg_slope1)
                slope_stage1_2.append(avg_slope2)
                slope_stage1_3.append(avg_slope3)
                intercept_stage1.append(avg_intercept)
            else:
                slopes1[iter][t-1][n] = avg_slope1 
                slopes2[iter][t-1][n] = avg_slope2  
                slopes3[iter][t-1][n] = avg_slope3
                intercepts[iter][t-1][n] = avg_intercept  

    iter += 1

end = time.process_time()
print('********************************************')
print('final expected total costs is %.2f' % -z)
print('ordering Q1 and Q2 in the first peiod is %.2f and %.2f' % (q1_values[iter-1][0][0], q2_values[iter-1][0][0]))
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time) 