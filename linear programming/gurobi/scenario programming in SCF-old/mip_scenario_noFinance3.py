# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 15:35:55 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: read from lp or mps file
    
"""
from memory_profiler import profile
import sys
from gurobipy import *


@profile
def lp():
    model = read('scenario_noFinance.mps')
    model.params.Method = 1
    model.params.Threads = 1
    model.optimize()
    
    if model.status == GRB.Status.INF_OR_UNBD:
        # Turn presolve off to determine whether model is infeasible
        # or unbounded
        model.setParam(GRB.Param.Presolve, 0)
        model.optimize()
    
    if model.status == GRB.Status.OPTIMAL:
        print('Optimal objective: %g' % model.objVal)
        model.write('model.sol')
        exit(0)
    elif model.status != GRB.Status.INFEASIBLE:
        print('Optimization was stopped with status %d' % model.status)
        exit(0)
    
    
    # Model is infeasible - compute an Irreducible Inconsistent Subsystem (IIS)
    
    print('')
    print('Model is infeasible')
    model.computeIIS()
    model.write("model.ilp")
    print("IIS written to file 'model.ilp'")
    
lp()
