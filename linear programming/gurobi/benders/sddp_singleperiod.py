# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 10:57:33 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: test sddp in a single period news vender problem:
second stage decision variable is selling quantity y
    
"""


import numpy as np
import scipy.stats as st
from gurobipy import *
import math
import random


# generate latin hypercube samples 
def generate_sample(sample_num, trunQuantile, mu):
    samples = [0 for i in range(sample_num)]
    for i in range(sample_num):
        rand_p = np.random.uniform(trunQuantile*i/sample_num, trunQuantile*(i+1)/sample_num)
        samples[i] = st.poisson.ppf(rand_p, mu)
    return samples


price  =  6
vari_cost = 2
sal_value = 1
mean_demands = 10
sample_num = 100
trunQuantile = 0.9999 # affective to the final ordering quantity

samples = generate_sample(sample_num, trunQuantile, mean_demands)
   
ini_Q = 10
ini_Obj = float("inf") # largest float number

## compute seconde stage objective in each scenario
N = len(samples)

k = 1
feasible_cut_index = []
Q = ini_Q
T= [1, 0]
obj = [0.0  for i in range(N)]
g = [0.0 for i in range(N)]
pi1 = [0.0  for i in range(N)]
Dpi2 = [0.0 for i in range(N)]

m = Model()
nita_value = -1000
nita = m.addVar(lb = nita_value , vtype = GRB.CONTINUOUS) # objective of the second stage
#m.addConstr(nita >= nita_value)
x = m.addVar(vtype = GRB.CONTINUOUS)
this_obj = vari_cost*x + nita
m.setObjective(this_obj, GRB.MINIMIZE)

while True:
    last_nita = nita_value
    last_master_obj = nita_value + vari_cost*Q
    for i in range(N):
        m1 = Model()    
        y = m1.addVar( vtype = GRB.CONTINUOUS)
        this_s_obj = -price*y - sal_value*(Q - y) # 
#        this_s_obj = -price*(Q-y) - sal_value*y # y is inventory
        m1.setObjective(this_s_obj, GRB.MINIMIZE)
        
        m1.addConstr(-y >= -Q) # should be standard model, or else the dual variable will be the opposite number
        m1.addConstr(-y >= -samples[i])
#        m1.addConstr(y >= Q - samples[i]) # can be wrong using min or max directly
#        m1.addConstr(y >= 0)
        # m1.addConstr(y == max(0, Q-samples[i])) # do not output the right resuls
        m1.write('test.lp')
        m1.optimize()       
        y_value = y.x
        
        obj[i] = m1.objVal
        pi = m1.getAttr(GRB.Attr.Pi) # important
        g[i] = sum([a*b for a,b in zip(pi, T)])
        pi1[i] = pi[0] # must be the dual variable of this constraint
        Dpi2[i] = pi[1] * samples[i]
        print()    
   
    avg_obj = sum(obj)/N    
    M = round(0.5*N)
    obj2 = random.choices(obj, k=M)
    avg_obj2 = sum(obj2)/N 
    var_obj2 = sum([(obj2[i] - avg_obj2)**2 for i in range(M)]) / (M-1)
    std_obj2 = math.sqrt(var_obj2)
    avg_g = sum(g)/N
    avg_pi1 = sum(pi1)/N
    avg_Dpi2 = sum(Dpi2)/N
#    m.addConstr(nita >= avg_obj - avg_pi1*(x-Q))
    avg_d = sum(samples)/N
    # 下面的这个约束条件相当于大于等于子问题对偶的目标函数值的期望，子问题中 x 是已知的，
    # 添加到这个约束条件中是未知的决策变量
    m.addConstr(nita >= -avg_pi1*x - avg_Dpi2 - sal_value*x) # just the benders optimality cut, same as the above constraint
    m.update()
    m.write('test.lp')  
    m.optimize()
    
    Q = x.x
    nita_value = nita.x
    master_obj = m.objVal
    print()
    
    k = k + 1
    # if abs(nita_value - avg_obj) < 1e-2 and k > 4: 
    #     break

    lb = avg_obj2 - 1.96*std_obj2/math.sqrt(M)
    ub = avg_obj2 + 1.96*std_obj2/math.sqrt(M)    
    # print(nita_value)
    # print(lb)
    # print(ub)  
    # print(Q)
    # if nita_value >= lb and nita_value <= ub:
    #     break
    # if abs(ub-lb) < 0.8:
    #     break
    if abs(last_master_obj - master_obj) < 1e-2 and k > 4:  # 这个终止条件最好
        break


print('iteration steps are %d' % k)    
print('ordering quantity is %.2f' % Q)
print('expected profit is %.2f' % master_obj)




