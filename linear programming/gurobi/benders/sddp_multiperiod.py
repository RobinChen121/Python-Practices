# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:25:47 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: test sddp in multi period newsvendor problems
    
"""


import numpy as np
import scipy.stats as st
from gurobipy import *
import time
from functools import reduce
import itertools
import random


def generate_sample(sample_num, trunQuantile, mu):
    samples = [0 for i in range(sample_num)]
    for i in range(sample_num):
        rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
        samples[i] = st.poisson.ppf(rand_p, mu)
    return samples
        

ini_I = 0
ini_cash = 0
price  =  6
vari_cost = 2
sal_value = 1
uni_hold_cost = 0
mean_demands = [5, 5]
sample_nums = [10, 10]
T = len(mean_demands)
trunQuantile = 0.9999 # affective to the final ordering quantity
scenario_num = reduce(lambda x, y: x * y, sample_nums, 1)
N = scenario_num

# samples_detail is the detailed samples in each period
samples_detail = [[0 for i in range(sample_nums[t])] for t in range(T)] 
for t in range(T):
    samples_detail[t] = generate_sample(sample_nums[t], trunQuantile, mean_demands[t])
scenarios = list(itertools.product(*samples_detail)) 
samples= scenarios #random.sample(scenarios, N)

z_ub = GRB.INFINITY
z_lb = -GRB.INFINITY

q = 5 # initial ordering quantity for stage 1
Q_ub = 2000 # 第二阶段的最优值，初始默认值
q_s = [[5 for t in range(T-1)] for i in range(N)] # initial q for stage 2 to T

I = [[0.0 for t in range(T)] for i in range(N)]
cash = [[0.0 for t in range(T)] for i in range(N)]
models = [[Model() for t in range(T)] for i in range(N)] # linear master models for stage 1 to T
ExpectQ = [[models[i][t].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS) for t in range(T-1)] for i in range(N)] # expected Q for stage 1 to T-1
x = [[models[i][t].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS) for t in range(T-1)] for i in range(N)] # order quantity for stage 2 to T
y = [[models[i][t].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS) for t in range(T)] for i in range(N)]
for i in range(N):
    for t in range(T):
        if t == 0:    
            I[i][t] = max(0, ini_I + q - scenarios[i][t])
            cash[i][t] = ini_cash + price*min(ini_I + q, samples[i][t]) - uni_hold_cost*I[i][t] - vari_cost*q
        else:
            I[i][t] = max(0, I[i][t-1] + q_s[i][t-1] - samples[i][t])
            cash[i][t] = cash[i][t-1] + price*min(I[i][t-1] + q_s[i][t-1] , samples[i][t]) - uni_hold_cost*I[i][t] - vari_cost*q_s[i][t-1] 
        
        if t < T-1:
            models[i][t].setObjective(-vari_cost*x[i][t] + price*y[i][t] - uni_hold_cost*(I[i][t-1] + q_s[i][t-1] - y[i][t]) + ExpectQ[i][t], GRB.MAXIMIZE)
            models[i][t].addConstr(ExpectQ[i][t] <= Q_ub)
        elif t == T-1 and t > 0:
            models[i][t].setObjective(price*y[i][t] + (sal_value - uni_hold_cost)*(I[i][t-1] + q_s[i][t-1]- y[i][t]), GRB.MAXIMIZE)
        else: # t==0 
            models[i][t].setObjective(price*y[i][t] + (sal_value - uni_hold_cost)*(ini_I + q - y[i][t]), GRB.MAXIMIZE)    
            
        if t == 0:
            models[i][t].addConstr(y[i][t] <= ini_I + q)
        else:
            models[i][t].addConstr(y[i][t] <= I[i][t-1] + q_s[i][t-1])
        models[i][t].addConstr(y[i][t] <= scenarios[i][t])
            
# initial model settings and cuts
main_m = Model() # linear model at the beginning of stage 1
ExpectQ_values = [[0.0 for t in range(T-1)] for i in range(N)]
main_x = main_m.addVar(vtype = GRB.CONTINUOUS)
mainExpectQ = main_m.addVar(vtype = GRB.CONTINUOUS)
main_m.addConstr(mainExpectQ <= Q_ub)
main_obj = -vari_cost*main_x + mainExpectQ
main_m.setObjective(main_obj, GRB.MAXIMIZE)

# 第一阶段期初的决策是x，之后每个阶段的决策变量是 y, x，T+1 阶段期初的决策变量是y
k = 1
obj_value = 3000 # initial objective value    
while True:
    last_obj_value = obj_value          

    for t in range(T-1, -1, -1):
        obj = [0.0  for i in range(N)]
        g = [0.0 for i in range(N)]
        for i in range(N):
            if k > 1: # update objective and some constraints that has q
                if t < T-1:
                    models[i][t].setObjective(-vari_cost*x[i][t] + price*y[i][t] - uni_hold_cost*(I[i][t-1] + q_s[i][t] - y[i][t]) + ExpectQ[i][t], GRB.MAXIMIZE) 
                elif t == T-1 and t > 0:
                    models[i][t].setObjective(price*y[i][t] + (sal_value - uni_hold_cost)*(I[i][t-1] + q_s[i][t]- y[i][t]), GRB.MAXIMIZE)
                else: # t==0 
                    models[i][t].setObjective(price*y[i][t] + (sal_value - uni_hold_cost)*(ini_I + q - y[i][t]), GRB.MAXIMIZE)
                    
                c = models[i][t].getConstrs()[0]
                if t == 0:
                    c.RHS = ini_I + q
                else:
                    c.RHS = I[i][t-1] + q_s[i][t]
                # models[i][t].remove(models[i][t].getConstrs()[0])
                # if t == 0:
                #     models[i][t].addConstr(y[i][t] <= ini_I + q)
                # else:
                #     models[i][t].addConstr(y[i][t] <= I[i][t-1] + q_s[i][t])
                models[i][t].update()            
            
            models[i][t].Params.LogToConsole = 0           
            models[i][t].optimize()                   
            obj[i] = models[i][t].objVal
            pi = models[i][t].getAttr(GRB.Attr.Pi)
            g[i] = pi[0]*1
            y_value = y[i][t].x
  
        avg_obj = sum(obj)/N
        avg_g = sum(g)/N
        if t > 0:
            models[i][t-1].addConstr(ExpectQ[i][t-1] >= avg_obj + avg_g*(x[i][t-1] - q_s[i][t-1])) # add cut
            models[i][t-1].update()
            # models[i][t-1].Params.LogToConsole = 0
            models[i][t-1].write('model.lp')
            models[i][t-1].optimize()

            ExpectQ_values[i][t-1] = ExpectQ[i][t-1].x
            q_s[i][t-1] = x[i][t].x
        else:
            main_m.addConstr(mainExpectQ <= avg_obj + avg_g*(main_x - q)) # add cut
            main_m.update()
            # main_m.Params.LogToConsole = 0
            main_m.optimize()   
             
            obj_value = main_m.objVal
            q = main_x.x
            print('')
    
    k = k + 1
    if abs(obj_value - last_obj_value) < 1e-1 and k > 4: # same effect with the above line
        break
    
print('iteration steps are %d' % k)    
print()
print('ordering quantity is %.2f' % main_x.x)
print('expected profit is %.2f' % obj_value)
    
