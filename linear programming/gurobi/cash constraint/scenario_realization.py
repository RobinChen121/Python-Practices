# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:03:18 2019

@author: zhen chen

MIT Licence.

Python version: 3.7

Email: robinchen@swu.edu.cn

Description: this a python test code for getting the demand realizaions in the scenario tree. This 
method is adopted from the paper Hu and Hu (2016) in IJPE

It shows that Gurobi can not solve nonlinear optimization problems when their degrees are higher than 2
    
"""

from gurobipy import *

# simulate the Weibull distribution in demand realizations
# mean = 467.25
# variance = 99.42
# skewness = 1.06
# kurosis = 4.35

# Create a new model
m = Model("scenario_realization")

realization_num = 5
mean = 467.25
variance = 99.42
skewness = 1.06
kurosis = 4.35

# Create variables: addVar is better than addVars() for its convenience of supporting computing
p = {} # possibilities
d = {} # demand realization values
for i in range(realization_num):
    p_name = 'p' + str(i)
    p[i] = m.addVar(name = p_name)
    d_name = 'd' + str(i)
    d[i] = m.addVar(name = d_name)

# objuective function: min (p*d - mean)^2
sample_mean = None
for i in range(realization_num):
    sample_mean = sample_mean + p[i] * d[i]
    
obj = (sample_mean - mean) * (sample_mean - mean)
m.setObjective(obj)

# constraint
m.addConstr(sum(p) == 1)

# solve and output
m.optimize()


