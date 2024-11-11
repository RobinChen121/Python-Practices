#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:59:57 2024

@author: zhenchen

@disp:  SDDP result is 167.02 for stationary, 3.29s; 214.87 for non-stationary, 3.29s 
    
    
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
mean_demands =[30, 30, 30, 30, 30] # higher average demand vs lower average demand
# betas = [2, 0.25] # lower variance vs higher variance
# T = len(mean_demands1)

pk = [0.25, 0.5, 0.25]
xk = [mean_demands[0]-10, mean_demands[0], mean_demands[0]+10]

cov = 0.25 # lower variance vs higher variance
sigmas = [cov*i for i in mean_demands]

mean_demands =  [10, 20, 10, 20] # [15, 15, 15, 15]
T = len(mean_demands)
if T == 4:
    opt = 215.48 #167.31 # 215.48 for non stationary
elif T == 3:
    opt = 26.68

T = len(mean_demands)

ini_I = 0
ini_cash = 0
vari_cost = 1
price = 10 # lower margin vs higher margin
unit_salvage = 0.5* vari_cost
overhead_cost = [50 for t in range(T)] # 100 for multi product

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
sample_details = [[0 for i in range(sample_num)] for t in range(T)]
for t in range(T):
    # sample_details1[t] = generate_samples_gamma(sample_num, trunQuantile, mean_demands1[t], betas[0])
    sample_details[t] = generate_samples(sample_num, trunQuantile, mean_demands[t])
    # sample_details[t] = generate_samples_normal(sample_num, trunQuantile, mean_demands[t], sigmas[t])
    # sample_details1[t] = generate_samples_discrete(sample_num, xk1, pk1)


# sample_details1 = [[10, 30], [10, 30], [10, 30]] # change 2


theta_iniValue = -1000 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q')
W0 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^0')
W1 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^1')
W2 = m.addVar(vtype = GRB.CONTINUOUS, name = 'w_1^2')
theta = m.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_2')

m.setObjective(overhead_cost[0] + vari_cost*q + r2*W2 + r1*W1 - r0*W0 + theta, GRB.MINIMIZE)
m.addConstr(theta >= theta_iniValue*(T))
m.addConstr(W1 <= U)
m.addConstr(-vari_cost*q - W0 + W1 + W2 == overhead_cost[0] - ini_cash)


# cuts recording arrays
iter_limit = 30
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
q_values = [] 
qpre_values = [] 
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
    slopes1.append([[0 for n in range(N)] for t in range(T)])
    slopes2.append([[0 for n in range(N)] for t in range(T)])
    slopes3.append([[0 for n in range(N)] for t in range(T)])
    intercepts.append([[0 for n in range(N)] for t in range(T-1)])
    q_values.append([[0 for n in range(N)] for t in range(T)]) 
    qpre_values.append([[0 for n in range(N)] for t in range(T)]) 
    
    cut_index[iter] = [0 for t in range(T-1)]
    cut_index_back[iter] = [0 for t in range(T-1)]
     
    # sample_scenarios1 = generate_scenario_samples_gamma(N, trunQuantile, mean_demands[0], betas[0], T)
    # sample_scenarios2 = generate_scenario_samples_gamma(N, trunQuantile, mean_demands[1], betas[1], T)
    
    sample_scenarios = generate_scenarios2(N, sample_num, sample_details)
    
    # sample_scenarios1 = generate_scenarios_discrete(N, xk1, pk1, T)
    # sample_scenarios2 = generate_scenarios_discrete(N, xk2, pk2, T)
    
    # sample_scenarios1 = generate_scenarios_normal(N, trunQuantile, mean_demands1, sigmas1)
    # sample_scenarios2 = generate_scenarios_normal(N, trunQuantile, mean_demands2, sigmas2)
     
    # sample_scenarios1 = [[10, 10, 10], [10,10, 30], [10, 30, 10], [10,30, 30],[30,10,10],[30,10,30],[30,30,10],[30,30,30]] # change 4
    # sample_scenarios2 = [[5, 5, 5], [5, 5, 15], [5, 15, 5], [5,15,15],[15,5,5], [15,5, 15], [15,15,5], [15,15,15]]
    
    # forward
    if iter > 0:        
        m.addConstr(theta >= slope_stage1_1[-1]*ini_I \
                            + slope_stage1_2[-1]*(ini_cash-vari_cost*q-r1*W1+r0*W0-r2*W2)\
                            + slope_stage1_3[-1]*q + intercept_stage1[-1])        
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    
    # if iter >= 0:
    #     m.write('iter' + str(iter+1) + '_main.lp')    
    #     m.write('iter' + str(iter+1) + '_main.sol')
    #     pass

    q_values[iter][0] = [q.x for n in range(N)]     
    W0_values.append(W0.x)
    W1_values.append(W1.x)
    W2_values.append(W2.x)
    z = m.objVal
    z_values = [[ 0 for t in range(T+1)] for n in range(N)] # for computing the feasible cost
    for n in range(N):
        z_values[n][0] = m.objVal - theta.x
      
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    qpre_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'qpre_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T-1)]
    I_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    cash_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1)+ '^' + str(n+1)) for n in range(N)] for t in range(T)]
    W0_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W1_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    W2_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    # B is the quantity of lost sale
    B_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
 
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    W0_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    W1_forward_values = [[0 for n in range(N)] for t in range(T-1)]
    W2_forward_values = [[0 for n in range(N)] for t in range(T-1)] 
    
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]
            
            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] + price*B_forward[t][n] == ini_cash - overhead_cost[t]\
                                          - vari_cost*q_values[-1][t][n]  -r1*W1_values[-1] + r0*W0_values[-1]\
                                              -r2*W2_values[-1] + price*demand)
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] + qpre_values[-1][t-1][n] - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] + price*B_forward[t][n]  == cash_forward_values[t-1][n] - overhead_cost[t]\
                                          - vari_cost*q_values[-1][t][n] -r1*W1_forward_values[t-1][n] + r0*W0_forward_values[t-1][n]\
                                              -r2*W2_forward_values[t-1][n] + price*demand)
             
            if t < T - 1:
                m_forward[t][n].addConstr(qpre_forward[t][n] == q_values[-1][t][n]) 
            if t == T - 1:                   
                m_forward[t][n].setObjective(-price*(demand - B_forward[t][n])\
                                             - unit_salvage*I_forward[t][n], GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(overhead_cost[t] + vari_cost*q_forward[t][n]\
                                             - price*(demand - B_forward[t][n])\
                                             + r2*W2_forward[t][n]\
                                             + r1*W1_forward[t][n] - r0*W0_forward[t][n] + theta_forward[t][n], GRB.MINIMIZE)  
                    
                m_forward[t][n].addConstr(W1_forward[t][n] <= U) 
                m_forward[t][n].addConstr(cash_forward[t][n] - vari_cost*q_forward[t][n] - W0_forward[t][n]\
                                          + W1_forward[t][n] + W2_forward[t][n] == overhead_cost[t+1]) 
                m_forward[t][n].addConstr(theta_forward[t][n] >= theta_iniValue*(T-1-t))                  
            
            # put those cuts in the back
            if iter > 0 and t < T - 1:
                for i in range(iter-1):
                    nn = cut_index[i][t] # i is the iter index
                    m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[i][t][nn]*(I_forward[t][n]+ qpre_forward[t][n])\
                                                      + slopes3[i][t][nn]*q_forward[t][n]\
                                                  + slopes2[i][t][nn]*(cash_forward[t][n]- vari_cost*q_forward[t][n]-r2*W2_forward[t][n]\
                                             - r1*W1_forward[t][n]+r0*W0_forward[t][n])\
                                                + intercepts[i][t][nn])
            NewJustAdd = True
            loop_no = 0 # necessary
            while NewJustAdd:      
                NewJustAdd = False            
                if iter > 0 and t < T - 1: 
                    nn = cut_index[iter-1][t] # default initiallly select the first cut
                    m_forward[t][n].addConstr(theta_forward[t][n] >= slopes1[iter-1][t][nn]*(I_forward[t][n]+ qpre_forward[t][n])\
                                                  + slopes3[iter-1][t][nn]*q_forward[t][n] +\
                                              + slopes2[iter-1][t][nn]*(cash_forward[t][n]- vari_cost*q_forward[t][n]-r2*W2_forward[t][n]\
                                         - r1*W1_forward[t][n]+r0*W0_forward[t][n])\
                                            + intercepts[iter-1][t][nn])
                    
                # optimize
                m_forward[t][n].Params.LogToConsole = 0
                m_forward[t][n].optimize()
                # if iter == 1 and t == 0:
                #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '.lp') 
                #     m_forward[t][n].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '.sol') 
                #     pass
            
                I_forward_values[t][n] = I_forward[t][n].x 
                               
                B_forward_values[t][n] = B_forward[t][n].x  
                cash_forward_values[t][n] = cash_forward[t][n].x 
        
                if t < T - 1: # for computing confidence interval
                    z_values[n][t+1] = m_forward[t][n].objVal - theta_forward[t][n].x
                else:
                    z_values[n][t+1] = m_forward[t][n].objVal
                                        
                if t < T - 1:
                    q_values[iter][t+1][n] = q_forward[t][n].x
                    qpre_values[iter][t][n] = qpre_forward[t][n].x
                    W1_forward_values[t][n] = W1_forward[t][n].x
                    W0_forward_values[t][n] = W0_forward[t][n].x
                    W2_forward_values[t][n] = W2_forward[t][n].x
                
                if iter > 0 and t < T - 1:                    
                    N2 = kk[iter-1] if iter <= len(kk) else kk[-1]
                    values = [0 for k in range(N2)]     
                    for k in range(N2):
                        values[k] = slopes1[iter-1][t][k]*(I_forward[t][n].x + qpre_forward[t][n].x)\
                                                      + slopes3[iter-1][t][k]*q_forward[t][n].x +\
                                                  + slopes2[iter-1][t][k]*(cash_forward[t][n].x- vari_cost*q_forward[t][n].x-r2*W2_forward[t][n].x\
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
    q_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T - 1)] 
    qpre_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'qpre_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T-1)] 
    
    I_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T)]
   
    cash_backward = [[[m_backward[t][n][s].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T)]    
    W0_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W0_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T - 1)] 
    W1_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W1_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T - 1)] 
    W2_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'W2_' + str(t+2) + '^' + str(n+1)) for s in range(sample_num)]  for n in range(N)] for t in range(T - 1)] 
    
    # B is the quantity of lost sale
    B_backward = [[[m_backward[t][n][s].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for s in range(sample_num)] for n in range(N)] for t in range(T)]
    theta_backward = [[[m_backward[t][n][s].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for s in range(sample_num)] for n in range(N)] for t in range(T - 1)]
    
    intercept_values = [[[0 for s in range(sample_num)] for n in range(N)] for t in range(T)]
    slope1_values = [[[0 for s in range(sample_num)] for n in range(N)] for t in range(T)] 
    slope2_values = [[[0 for s in range(sample_num)] for n in range(N)] for t in range(T)] 
    slope3_values = [[[0 for s in range(sample_num)] for n in range(N)] for t in range(T)]
    
    for t in range(T-1, -1, -1):  
        for n in range(N):      
            S = sample_num # should revise, should be S^2
            IpW0 = False
            IpW1 = False
            IpW2 = False
            InW0 = False
            InW1 = False
            InW2 = False
            IpW0_values = []
            IpW1_values = []
            IpW2_values = []
            InW0_values = []
            InW1_values = []
            InW2_values = []
            StateChange = False
            for s in range(S):
                demand = sample_details[t][s]  
                thisEndCash = 0.0
                if s > 0:
                    if t == 0:
                        thisEndI = ini_I - demand
                        thisB = -min(ini_I - demand, 0)
                        if t < T - 1:
                            thisEndCash = ini_cash - overhead_cost[t] - vari_cost*q_values[-1][t][n]\
                                                      - r2*W2_values[-1] - r1*W1_values[-1] + r0*W0_values[-1] + price*(demand - thisB) - vari_cost*lastq - overhead_cost[t+1]
                    else:
                        thisEndI = I_forward_values[t-1][n] + qpre_values[-1][t-1][n] - demand
                        thisB = -min(I_forward_values[t-1][n] + qpre_values[-1][t-1][n] - demand, 0)
                        if t < T - 1:
                            thisEndCash = cash_forward_values[t-1][n] - overhead_cost[t-1] - vari_cost*q_values[-1][t][n]\
                                                      - r2*W2_forward_values[t-1][n]- r1*W1_forward_values[t-1][n] + r0*W0_forward_values[t-1][n] + price*(demand - thisB) - vari_cost*lastq - overhead_cost[t+1]
                    if thisEndI >= 0 and thisEndCash >= 0:
                        if len(IpW0_values ) > 0:
                            pi, rhs = IpW0_values 
                        else:
                            StateChange = True
                    elif thisEndI >= 0 and 0 > thisEndCash >= -U :
                        if len(IpW1_values ) > 0:
                            pi, rhs = IpW1_values 
                        else:
                            StateChange = True
                    elif thisEndI >= 0 and thisEndCash < -U :
                        if len(IpW2_values ) > 0:
                            pi, rhs = IpW2_values 
                        else:
                            StateChange = True                   
                    elif thisEndI < 0 and thisEndCash >= 0:
                        if len(InW0_values ) > 0:
                            pi, rhs = InW0_values 
                        else:
                            StateChange = True
                    elif thisEndI < 0 and 0 > thisEndCash >= -U:
                        if len(InW1_values ) > 0:
                            pi, rhs = InW1_values 
                        else:
                            StateChange = True
                    elif thisEndI < 0 and thisEndCash < -U:
                        if len(InW2_values ) > 0:
                            pi, rhs = InW2_values 
                        else:
                            StateChange = True
                if s == 0 or StateChange == True:
                    if t == T - 1:                   
                        m_backward[t][n][s].setObjective(-price*(demand - B_backward[t][n][s])\
                                                         - unit_salvage*I_backward[t][n][s], GRB.MINIMIZE)
                    else:
                        m_backward[t][n][s].setObjective(overhead_cost[t] + vari_cost*q_backward[t][n][s] \
                                                         - price*(demand - B_backward[t][n][s])\
                                                         + r2*W2_backward[t][n][s]
                                                         + r1*W1_backward[t][n][s] - r0*W0_backward[t][n][s] + theta_backward[t][n][s], GRB.MINIMIZE)  
                    if t == 0:   
                        m_backward[t][n][s].addConstr(I_backward[t][n][s] - B_backward[t][n][s] == ini_I - demand)
                        m_backward[t][n][s].addConstr(cash_backward[t][n][s] == ini_cash - overhead_cost[t] - vari_cost*q_values[-1][t][n]\
                                                      - r2*W2_values[-1] - r1*W1_values[-1] + r0*W0_values[-1]\
                                                      + price*(demand - B_backward[t][n][s]))
                    else:
                        m_backward[t][n][s].addConstr(I_backward[t][n][s] - B_backward[t][n][s] == I_forward_values[t-1][n] + qpre_values[-1][t-1][n] - demand)
                        m_backward[t][n][s].addConstr(cash_backward[t][n][s] + price*B_backward[t][n][s] == cash_forward_values[t-1][n]- overhead_cost[t]\
                                                      - vari_cost*q_values[-1][t][n]\
                                                     - r2*W2_forward_values[t-1][n]- r1*W1_forward_values[t-1][n]\
                                                      + r0*W0_forward_values[t-1][n] + price*demand)
                
                    if t < T - 1:
                        m_backward[t][n][s].addConstr(qpre_backward[t][n][s] == q_values[-1][t][n]) 
                    if t < T - 1:                   
                        m_backward[t][n][s].addConstr(W1_backward[t][n][s] <= U)
                        m_backward[t][n][s].addConstr(cash_backward[t][n][s] -vari_cost*q_backward[t][n][s] - W0_backward[t][n][s]\
                                                      + W1_backward[t][n][s] + W2_backward[t][n][s] == overhead_cost[t+1])
                        m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= theta_iniValue*(T-1-t))
                        
                    # put those cuts in the back
                    if iter > 0 and t < T - 1:
                        for i in range(iter - 1):
                            nn = cut_index_back[i][t]
                            m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= slopes1[i][t][nn]*(I_backward[t][n][s]+ qpre_backward[t][n][s])\
                                                          + slopes3[i][t][nn]*q_backward[t][n][s]+\
                                                          + slopes2[i][t][nn]*(cash_backward[t][n][s]- vari_cost*q_backward[t][n][s]\
                                                                               -r2*W2_backward[t][n][s]- r1*W1_backward[t][n][s]+r0*W0_backward[t][n][s])\
                                                        + intercepts[i][t][nn])
                    
                    NewJustAdd = True
                    loop_no = 0
                    while NewJustAdd:
                        NewJustAdd = False
                        if iter > 0 and t < T - 1:
                            nn = cut_index_back[iter - 1][t]
                            m_backward[t][n][s].addConstr(theta_backward[t][n][s] >= slopes1[iter - 1][t][nn]*(I_backward[t][n][s]+ qpre_backward[t][n][s])\
                                                          + slopes3[iter - 1][t][nn]*q_backward[t][n][s]\
                                                          + slopes2[iter - 1][t][nn]*(cash_backward[t][n][s]- vari_cost*q_backward[t][n][s]\
                                                                               -r2*W2_backward[t][n][s]- r1*W1_backward[t][n][s]+r0*W0_backward[t][n][s])\
                                                        + intercepts[iter - 1][t][nn])
                        # optimize
                        m_backward[t][n][s].Params.LogToConsole = 0
                        m_backward[t][n][s].optimize()                  
                        pi = m_backward[t][n][s].getAttr(GRB.Attr.Pi)
                        rhs = m_backward[t][n][s].getAttr(GRB.Attr.RHS)
                        
                        if iter > 0 and t < T - 1:                    
                            N2 = kk[iter-1] if iter <= len(kk) else kk[-1]
                            values = [0 for k in range(N2)]     
                            for k in range(N2):
                                values[k] = slopes1[iter-1][t][k]*(I_backward[t][n][s].x + qpre_backward[t][n][s].x)\
                                                              + slopes3[iter-1][t][k]*q_backward[t][n][s].x +\
                                                          + slopes2[iter-1][t][k]*(cash_backward[t][n][s].x - vari_cost*q_backward[t][n][s].x - r2*W2_backward[t][n][s].x\
                                                     - r1*W1_backward[t][n][s].x + r0*W0_backward[t][n][s].x)\
                                                        + intercepts[iter-1][t][k]
                            if np.argmax(values) != nn:                       
                                NewJustAdd = True
                                loop_no += 1
                                if loop_no > 3:
                                    break
                                cut_index[iter - 1][t] = np.argmax(values)
                                m_backward[t][n][s].remove(m_backward[t][n][s].getConstrs()[-1])
                
                        if t < T - 1:
                            lastq = q_backward[t][n][s].X
                            thisEndCash = cash_backward[t][n][s].X - vari_cost*lastq - overhead_cost[t+1] 
                        if I_backward[t][n][s].X > 0:
                            thisEndI = I_backward[t][n][s].X
                        if B_backward[t][n][s].X > 0:
                            thisEndI = -B_backward[t][n][s].X                                    
                        if thisEndI >= 0 and thisEndCash >= 0:
                            IpW0_values = [pi, rhs]
                        elif thisEndI >= 0 and 0 > thisEndCash >= -U :
                            IpW1_values = [pi, rhs]
                        elif thisEndI >= 0 and thisEndCash < -U :
                            IpW2_values = [pi, rhs]                      
                        elif thisEndI < 0 and thisEndCash >= 0:
                            InW0_values = [pi, rhs]
                        elif thisEndI < 0 and 0 > thisEndCash >= -U:
                            InW1_values = [pi, rhs]
                        elif thisEndI < 0 and thisEndCash < -U:
                            InW2_values = [pi, rhs]
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
                    intercept_values[t][n][s] += -pi[0]*demand + pi[1]*price*demand\
                                                 - pi[1]*overhead_cost[t]- price*demand+ overhead_cost[t+1] # + pi[3]*U + pi[4]*overhead_cost[t+1]-pi[5]*theta_iniValue*(T-1-t) 
                else:
                    intercept_values[t][n][s] += -pi[0]*demand + pi[1]*price*demand\
                                                 - pi[1]*overhead_cost[t] - price*demand
                for sk in range(3, num_con):
                    intercept_values[t][n][s] += pi[sk]*rhs[sk]
                
                slope1_values[t][n][s] = pi[0]                                                        
                slope2_values[t][n][s] = pi[1]
                if t < T -1:
                    slope3_values[t][n][s] = pi[2]
                
            avg_intercept = sum(intercept_values[t][n]) / S
            avg_slope1 = sum(slope1_values[t][n]) / S
            avg_slope2 = sum(slope2_values[t][n]) / S
            avg_slope3 = sum(slope3_values[t][n]) / S
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
print('cut selection')
final_value = -z
print('sample numer is %d and scenario number is %d ' % (sample_num, N))
print('planning horizon length is T = %d ' % T)
print('final expected total profits after %d iteration is %.2f' % (iter, -z))
print('ordering Q in the first peiod is %.2f' % (q_values[iter-1][0][0]))
gap = (-opt + final_value)/opt               
print('optimaility gap is %.2f%%' % (100*gap))  
cpu_time = end - start
print('cpu time is %.3f s' % cpu_time) 
z_lb, z_ub, z_mean = compute_ub(z_values) # for computing confidence interval
lb = -np.mean(np.sum(z_values, axis=1))
print('expected lower bound gap is %.2f' % lb)  
gap2 = abs((z+lb)/z)
print('lower bound and upper bound gap is %.2f%%' % (100*gap2))  
print('confidence interval for expected objective is [%.2f,  %.2f]' % (-z_ub, -z_lb))  
