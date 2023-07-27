#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:52:47 2023

@author: zhenchen

@disp:  
    
    
"""

from gurobipy import *
import itertools
import random
import time

import sys 
sys.path.append("..") 
from tree import generate_sample, get_tree_strcture, generate_scenario_samples



start = time.process_time()
ini_I = 0
ini_cash = 10
vari_cost = 1
price = 10
unit_back_cost = 0
unit_hold_cost = 0
mean_demands = [10, 10]
T = len(mean_demands)
sample_nums = [10 for t in range(T)]

trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_numTotal = 1
for i in sample_nums:
    scenario_numTotal *= i

# detailed samples in each period
sample_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    sample_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
# samples_detail = [[5, 15], [5, 15]]
scenarios_full = list(itertools.product(*sample_detail)) 


iter = 1
iter_num = 15
N = 20 # sampled number of scenarios for forward computing

theta_iniValue = -300 # initial theta values (profit) in each period
m = Model() # linear model in the first stage
# decision variable in the first stage model
q = m.addVar(vtype = GRB.CONTINUOUS, name = 'q_1')
theta = m.addVar(lb = theta_iniValue*T, vtype = GRB.CONTINUOUS, name = 'theta_2')
m.setObjective(vari_cost*q + theta, GRB.MINIMIZE)
m.addConstr(q <= capacity)
m.addConstr(vari_cost * q <= ini_cash)

q_value = 0
theta_value = 0

# cuts
slope1_stage = []
intercept1_stage = []
slopes = [[ [] for n in range(N)] for t in range(T-1)]
intercepts = [[ [] for n in range(N)] for t in range(T-1)]
q_values = [0 for iter in range(iter_num)]
q_sub_values = [[[0 for n in range(N)] for t in range(T-1)] for iter in range(iter_num)]

start = time.process_time()
while iter <= iter_num:  
    
    # sample a numer of scenarios from the full scenario tree
    # random.seed(10000)
    sample_scenarios = generate_scenario_samples(N, trunQuantile, mean_demands)
    sample_scenarios.sort() # sort to make same numbers together
    
    # forward
    if iter > 0:
        m.addConstr(theta >= slope1_stage[-1]*q + intercept1_stage[-1])
    m.update()
    m.optimize()
    # m.write('iter' + str(iter) + '_main2.lp')    
    # m.write('iter' + str(iter) + '_main2.sol')
    
    q_values[iter] = q.x
    theta_value = theta.x
    z = m.objVal
    
    
    m_forward = [[Model() for n in range(N)] for t in range(T)]
    q_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]
    I_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    cash_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1)+ '^' + str(n+1)) for n in range(N)] for t in range(T)]
    
    # B is the quantity of lost sale
    B_forward = [[m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for n in range(N)]  for t in range(T)]
    theta_forward = [[m_forward[t][n].addVar(lb = -theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for n in range(N)]  for t in range(T - 1)]

    q_forward_values = [[0 for n in range(N)] for t in range(T)] 
    I_forward_values = [[0 for n in range(N)] for t in range(T)]
    B_forward_values = [[0 for n in range(N)] for t in range(T)]
    cash_forward_values = [[0 for n in range(N)] for t in range(T)]
    theta_forward_values = [[0 for n in range(N)] for t in range(T)]
    
    for t in range(T):
        for n in range(N):
            demand = sample_scenarios[n][t]

            
            if t == T - 1:                   
                m_forward[t][n].setObjective(-price*(demand - B_forward[t][n]), GRB.MINIMIZE)
            else:
                m_forward[t][n].setObjective(vari_cost*q_forward[t][n] - price*(demand - B_forward[t][n]) + theta_forward[t][n], GRB.MINIMIZE)  
                m_forward[n][t].addConstr(theta_forward[n][t] >= theta_iniValue*(T-1-t))
                m.addConstr(vari_cost * q_forward[t][n] <= cash_forward_values[t-1][n])
            if t == 0:   
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == ini_I + q_values[iter] - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] == ini_cash - vari_cost*q_forward[t][n] + price*(demand - B_forward[t][n]))
            else:
                m_forward[t][n].addConstr(I_forward[t][n] - B_forward[t][n] == I_forward_values[t-1][n] + q_forward_values[t][n] - demand)
                m_forward[t][n].addConstr(cash_forward[t][n] == cash_forward[t-1][n]- vari_cost*q_forward[t][n] + price*(demand - B_forward[t][n]))
                
            # optimize
            m_forward[n][t].optimize()
            I_forward_values[n][t] = I_forward[n][t].x 
            B_forward_values[n][t] = B_forward[n][t].x  
            cash_forward_values[n][t] = cash_forward[n][t].x 
            if t < T - 1:
                q_forward_values[n][t] = q_forward[n][t].x
                theta_forward_values[n][t] = theta_forward[n][t].x
    
    # backward
    m_backward = [[[Model() for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    q_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'q_' + str(t+2) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T - 1)] 
    I_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'I_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T)]
    cash_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'C_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])]  for n in range(N)] for t in range(T)]    
    # B is the quantity of lost sale
    B_backward = [[[m_backward[t][n][k].addVar(vtype = GRB.CONTINUOUS, name = 'B_' + str(t+1) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward = [[[m_backward[t][n][k].addVar(lb = -theta_iniValue*(T-1-t), vtype = GRB.CONTINUOUS, name = 'theta_' + str(t+3) + '^' + str(n+1)) for k in range(sample_nums[t])] for n in range(N)] for t in range(T - 1)]

    q_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    I_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    cash_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    B_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    theta_backward_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    pi_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)]
    pi_rhs_values = [[[0  for k in range(sample_nums[t])] for n in range(N)] for t in range(T)] 

    for t in range(T-1, -1, -1):    
        for n in range(N):      
            K = len(sample_detail[t])
            for k in range(K):
                demand = sample_scenarios[t][k]
                
                if t == T - 1:                   
                    m_backward[t][n][k].setObjective(-price*(demand - B_backward[t][n][k]), GRB.MINIMIZE)
                else:
                    m_backward[t][n][k].setObjective(vari_cost*q_backward[t][n][k] - price*(demand - B_backward[t][n][k]) + theta_backward[t][n][k], GRB.MINIMIZE)  
                    m_backward[n][t].addConstr(theta_backward[n][t] >= theta_iniValue*(T-1-t))
                    m.addConstr(vari_cost * q_backward[t][n][k] <= cash_forward_values[t-1][n])
                if t == 0:   
                    m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == ini_I + q_values[iter] - demand)
                    m_backward[t][n][k].addConstr(cash_backward[t][n][k] == ini_cash - vari_cost*q_backward[t][n][k] + price*(demand - B_backward[t][n][k]))
                else:
                    m_backward[t][n][k].addConstr(I_backward[t][n][k] - B_backward[t][n][k] == I_backward_values[t-1][n] + q_backward_values[t][n][k] - demand)
                    m_backward[t][n][k].addConstr(cash_backward[t][n][k] == cash_forward[t-1][n]- vari_cost*q_backward[t][n][k] + price*(demand - B_backward[t][n][k]))
                
                # optimize
                m_backward[t][n][k].optimize()                
                # if t == 0 and n == 0 and iter > 0:
                #     m_backward[t][n][k].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(k+1) +'-2back.lp')
                # if t > 0:
                #     m_backward[t][n][k].write('iter' + str(iter) + '_sub_' + str(t+1) + '^' + str(n+1) + '_' + str(k+1) +'-2back.lp')
                
                pi = m_backward[t][n][k].getAttr(GRB.Attr.Pi)
                rhs = m_backward[t][n][k].getAttr(GRB.Attr.RHS)
                if t < T - 1:
                    num_con = len(pi)
                    for kk in range(num_con-1):
                        pi_rhs_values[t][n][k] += pi[kk]*rhs[kk]
                    pi_rhs_values[t][n][k] += -pi[-1]*demand 
                else:
                    pi_rhs_values[t][n][k] = -pi[-1] * demand
                pi_values[t][n][k] = pi[-1]
            
            print()





