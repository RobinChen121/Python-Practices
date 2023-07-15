#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:50:58 2023

@author: zhenchen

@disp:  test the linear model with bounds in variables
    
    
"""



from gurobipy import *

try:

    # Create a new model
    m = Model()

    # Create variables
    x = m.addVar(vtype=GRB.CONTINUOUS, name="x")
    y = m.addVar(lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")
    z = m.addVar(vtype=GRB.CONTINUOUS, name="z")

    # Set objective
    m.setObjective(15 * y - 100 * z, GRB.MAXIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(10 * x <= 1, "c0")

    # Add constraint: x + y >= 1
    m.addConstr(10 * x + y <= 2, "c1")
    
    m.addConstr(-y <= 10, "c3")
    m.addConstr(x + z <= 1, "c4")

    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')

