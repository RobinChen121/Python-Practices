# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 12:14:04 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description:  for computing the upper bound and lower bound of saa 
    
"""

from gurobipy import *
from gurobipy import LinExpr
from gurobipy import GRB
from gurobipy import Model
import time
import numpy as np
import scipy.stats as st
from math import exp
import itertools
import csv
import _pickle as cPickle # save python list to files


def lognorm_ppf(x, mu, sigma):
    shape  = sigma
    loc    = 0
    scale  = exp(mu)
    return st.lognorm.ppf(x, shape, loc, scale)

def generate_sample(sample_num, trunQuantile, mus, sigmas, booming_demand):
    T = len(booming_demand)
    N = len(mus)
    samples = [[[0 for i in range(sample_num[t])] for n in range(N)] for t in range(T)]
    for t in range(T):
        for i in range(sample_num[t]):
            rand_p = np.random.uniform(trunQuantile*i/sample_num[t], trunQuantile*(i+1)/sample_num[t])
            for n in range(N):
                samples[t][n][i] = lognorm_ppf(rand_p, mus[n][booming_demand[t]], sigmas[n][booming_demand[t]])
    return samples

  
    
# parameter values
ini_I = [0, 0, 0]
# prices = [89, 159, 300]
# vari_costs = [70, 60, 60]
prices = [189, 144, 239]
vari_costs = [140, 70, 150]
ini_cash = 20000

T = 6
overhead_cost = [2000 for t in range(T)]
booming_demand = [0, 0, 0, 0, 1, 1]
delay_length = 2
discount_rate = 0.01
B = 10000  # total quantity of order loan
ro = 0.015  # loan rate
M = 10000

mus = [[3.66, 5.79], [4.13, 5.91], [3.54, 4.96]]
sigmas = [[0.6, 0.26], [0.66, 0.33], [0.46, 0.18]]
#mus = [[3.66, 5.79], [4.13, 5.91]]
#sigmas = [[0.6, 0.26], [0.66, 0.33]]
N = len(mus)
sample_nums = [5, 5, 5, 3, 3, 3]
trunQuantile = 0.9999

KK = 10 # running times

headers = ['run','Final Value','Q1_0', 'Q2_0', 'Q3_0', 'Time','loan-used percent']
rows = [[0 for i in range(7)] for j in range(KK)]
for iK in range(KK):
    samples = generate_sample(sample_nums, trunQuantile, mus, sigmas, booming_demand[0:T])
    file_name = 'data' + str(iK+11) + '.pkl'
    cPickle.dump(samples, open(file_name, "wb"))
    S = np.prod(sample_nums[0:T])
    arr = []
    for t in range(T):
        arr.append(range(sample_nums[t]))
    scenario_permulations = list(itertools.product(*arr))
    
    tic = time.time()
    try:
        # Create a new model
        m = Model("order-loan-saa")
    
        # Create variables
    #    Q0 = [m.addVar(vtype = GRB.CONTINUOUS) for n in range(N)]
        Q = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)]for n in range(N)] for t in range(T)]
        I = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for t in range(N)] for n in range(T)] # end-of-period inventory in each period for each product
        delta = [[[m.addVar(vtype = GRB.BINARY) for s in range(S)] for t in range(N)] for n in range(T)] # whether lost-sale not occurs
        g = [[[m.addVar(vtype = GRB.CONTINUOUS) for s in range(S)] for n in range(N)] for t in range(T)] # order-loan quantity in each period for each product
         
        C = [[LinExpr()  for s in range(S)] for t in range(T)] # LinExpr, end-of-period cash in each period
        R = [[[LinExpr()  for s in range(S)] for n in range(N)] for t in range(T + delay_length)]  # LinExpr, revenue for each product in each period
        
        # revenue expression  # check revenue
        for s in range(S):
            for n in range(N):
                for t in range(T + delay_length):
                    if t < delay_length:
                        R[t][n][s] = prices[n] * g[t][n][s]
                    else:
                        if t == delay_length:
                            R[t][n][s] = prices[n]*(ini_I[n]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
                        elif t < T:
                            R[t][n][s] = prices[n]*(g[t][n][s]+I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
                        else:        
                            R[t][n][s] = prices[n]*(I[t-delay_length-1][n][s]+Q[t-delay_length][n][s]-I[t-delay_length][n][s]-g[t-delay_length][n][s]-g[t-delay_length][n][s]*(1+ro)**delay_length)
        
        m.update()
        
        # cash flow   
        revenue_total = [[LinExpr() for s in range(S)] for t in range(T)]
        vari_costs_total = [[LinExpr() for s in range(S)] for t in range(T)]
        expect_revenue_total = [LinExpr() for t in range(T)]
        expect_vari_costs_total = [LinExpr() for t in range(T)]
        for s in range(S):
            for t in range(T):
                revenue_total[t][s] = sum([R[t][n][s] for n in range(N)])
                vari_costs_total[t][s] = sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) # strange
                try:
                    if t == 0:
                        C[t][s] = ini_cash + revenue_total[t][s] - vari_costs_total[t][s] - overhead_cost[t]
                    else:
                        C[t][s] = C[t-1][s] + revenue_total[t][s] - vari_costs_total[t][s]- overhead_cost[t]
                except:
                    print(n)   
        
        for t in range(T):
            expect_revenue_total[t] = sum([revenue_total[t][s] / S for s in range(S)])
            expect_vari_costs_total[t] = sum([vari_costs_total[t][s] / S for s in range(S)])
            
            
        m.update()
                
        # objective function          
        discounted_cash = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for k in range(delay_length):
                    discounted_cash[s] = discounted_cash[s] + R[T+k][n][s] / (1 + discount_rate)**(k+1)    
        final_cash = sum([(C[T-1][s] + discounted_cash[s])/ S for s in range(S)])
        expect_discounted_cash = sum([(discounted_cash[s])/ S for s in range(S)])
        
        
        # Add constraints
    #    for s in range(S):
    #        for n in range(N):
    #            for t in range(T):
    #                m.addConstr(Q0 == 0) 
        # inventory flow   
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    demand = samples[t][n][scenario_permulations[s][t]]  # be careful
                    if t == 0:
                        m.addConstr(I[t][n][s] <= ini_I[n] + Q[t][n][s] - demand + (1 - delta[t][n][s]) * M)     
                        m.addConstr(I[t][n][s] >= ini_I[n] + Q[t][n][s] - demand - (1 - delta[t][n][s]) * M)   
                        m.addConstr(ini_I[n] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                    else:
                        try:
                            m.addConstr(I[t][n][s] <= I[t-1][n][s]+ Q[t][n][s] - demand  + (1 - delta[t][n][s]) * M)     
                            m.addConstr(I[t][n][s] >= I[t-1][n][s] + Q[t][n][s] - demand  - (1 - delta[t][n][s]) * M)  
                            m.addConstr(I[t-1][n][s] + Q[t][n][s] - demand  <= delta[t][n][s]* M -0.1) 
                        except:
                            print(n)
                    m.addConstr(I[t][n][s] <= delta[t][n][s] * M)         
            
        # cash constraint
        for s in range(S):
            for t in range(T):
                if t == 0:
                    m.addConstr(ini_cash >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t]) # cash constaints
                else:       
                    m.addConstr(C[t - 1][s] >= sum([vari_costs[n] * Q[t][n][s] for n in range(N)]) + overhead_cost[t]) # cash constaints  
        
        # non-negavtivety of I_t
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    m.addConstr(I[t][n][s] >= 0)
        
        # order loan quantity less than realized demand
        # careful, there is no delay_length in this constraint
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    if t == 0:
                        m.addConstr(g[t][n][s] <= ini_I[n] + Q[t][n][s]-I[t][n][s])
                    else:
                        m.addConstr(g[t][n][s] <= I[t-1][n][s]+Q[t][n][s]-I[t][n][s])   
        # total order loan limit
        total_loan = [LinExpr() for s in range(S)]
        for s in range(S):
            for n in range(N):
                for t in range(T):
                    total_loan[s] += prices[n] * g[t][n][s]
        for s in range(S):
            m.addConstr(total_loan[s] <= B)
            
        # first-stage decision
        for s in range(S-1):
            for n in range(N):
                m.addConstr(Q[0][n][s] == Q[0][n][s+1])
                    
        
                        
        # Set objective
        m.update()
        m.setObjective(final_cash, GRB.MAXIMIZE)
                           
        # solve
        m.update()
        m.optimize()
        print('') 

        print('final expected value is: %g' % m.objVal)
        
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))    
            
    except AttributeError:
        print('Encountered an attribute error')
        
    toc = time.time()
    time_pass = toc - tic
    print('running time is %.2f' % time_pass)
    rows[iK] = [iK+1, m.objVal, Q[0][0][0].X, Q[0][1][0].X, Q[0][2][0].X, time_pass, percent]

with open('results-upperBound.csv','a', newline='') as f: # newline = '' is to remove the blank line
    f_csv = csv.writer(f)
#    f_csv.writerow(headers)
    f_csv.writerows(rows)