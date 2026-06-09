# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 10:57:33 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: test sddp in a single period news vender problem
    
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


price  =  10
vari_cost = 1
sal_value = 0
mean_demands = 10
sample_num = 100
trunQuantile = 0.9999 # affective to the final ordering quantity

samples = generate_sample(sample_num, trunQuantile, mean_demands)
   
ini_Q = 6
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

m2 = Model()
nita = m2.addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS) # objective of the second stage
nita_value = -3000
m2.addConstr(nita >= nita_value)
x = m2.addVar(vtype = GRB.CONTINUOUS)
this_obj = vari_cost*x + nita
m2.setObjective(this_obj, GRB.MINIMIZE)

while True:
    last_nita = nita_value
    last_master_obj = nita_value + vari_cost*Q
    for i in range(N):
        m = Model()    
        y = m.addVar(vtype = GRB.CONTINUOUS)
        this_s_obj = -price*y - sal_value*(Q - y) # or sal_value*(Q - y) , max(0, Q-samples[i])
        m.setObjective(this_s_obj, GRB.MINIMIZE)
        
        m.addConstr(-y >= -Q)
        m.addConstr(-y >= -samples[i])
        m.optimize()       
        y_value = y.x
        
        obj[i] = m.objVal
        pi = m.getAttr(GRB.Attr.Pi)
        g[i] = sum([a*b for a,b in zip(pi, T)])
        pi1[i] = -pi[0]
        Dpi2[i] = -pi[1] * samples[i]
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
    m2.addConstr(nita >= avg_obj - avg_g*(x-Q)) # add cut
    #m2.addConstr(nita >= avg_pi1*x + avg_Dpi2) # same as the above constraint
    m2.update()
    m2.optimize()
    
    Q = x.x
    nita_value = nita.x
    master_obj = m2.objVal
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
