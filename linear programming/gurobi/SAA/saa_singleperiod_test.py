# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:08:06 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: SAA for single period newsvendor, 
    
"""

import numpy as np
import scipy.stats as st
from gurobipy import *


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
mean_demands = 5
sample_num = 100
trunQuantile = 0.999

samples = generate_sample(sample_num, trunQuantile, mean_demands)
   

try:
    # Create a new model
    m = Model("saa-mip")
    
    # Create variables
    Q = m.addVar(vtype = GRB.CONTINUOUS) # only one value for all samples           
    Rd = [m.addVar(vtype = GRB.CONTINUOUS) for s in range(sample_num)] # realized demand, auxiliary variable
    remnant_demand = [m.addVar(vtype = GRB.CONTINUOUS) for s in range(sample_num)] # auxiliary variable
    delta = [m.addVar(vtype = GRB.BINARY) for s in range(sample_num)] # auxiliary variable
    profit = LinExpr() 
    M = 100000
    
    profit = price * sum(Rd) / sample_num - vari_cost * Q + sal_value * sum(remnant_demand) / sample_num
    
    m.update()
    m.setObjective(profit, GRB.MAXIMIZE)
    
    # Add constraints
    for s in range(sample_num):
        #m.addGenConstrMin(Rd[s], [Q], samples[s])
        # m.addConstr(Rd[s] == min_(Q, samples[s]))
        m.addConstr(Rd[s] >= samples[s] - delta[s] * M)
        m.addConstr(Rd[s] <= samples[s])
        m.addConstr(Rd[s] <= Q + (1 - delta[s]) * M)
        m.addConstr(Rd[s] >= Q - (1 - delta[s]) * M)
        m.addConstr(Q >= samples[s] - delta[s] * M)
        m.addConstr(remnant_demand[s] == Q - Rd[s])
        
    m.update()
    m.optimize()
    
    print('ordering quantity in the first period is:  %.2f ' % Q.X)
    
except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))    
        
except AttributeError:
        print('Encountered an attribute error')