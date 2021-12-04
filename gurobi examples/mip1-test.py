#!/usr/bin/python

# Copyright 2019, Gurobi Optimization, LLC

# This example formulates and solves the following simple MIP model:
#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

from gurobipy import *

try:

    # Create a new model
    m = Model("mip1")

    # Create variables
    x = m.addVar(vtype=GRB.CONTINUOUS, name="x") # default bounds for continuous type is [0, infinite]
    y = m.addVar(vtype=GRB.CONTINUOUS, name="y")
    z = m.addVar(vtype=GRB.CONTINUOUS, lb = -GRB.INFINITY, name="z")

    # # Set objective
    m.setObjective(-4*x+-7*y+z, GRB.MINIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(z >= -200, "c0")
    m.addConstr(11*x+19*y <= 42, "c1")
    m.addConstr(7/5*x +13/5* y -34/5 <= z, "c0")
    m.addConstr(y <= 2, "c0")
    m.addConstr(x <= 2, "c0")
    m.addConstr(y >= 2, "c0")
    m.addConstr(x >= 1, "c0")
    
    # #   #  Set objective
    # m.setObjective(-4*x+0*y-14, GRB.MAXIMIZE)

    # # Add constraint: x + 2 y + 3 z <= 4
    # m.addConstr(-4*x-2*y <= -2, "c0")
    # m.addConstr(2*x-3*y <= 3, "c1")
    # m.addConstr(-3*x + y <= -1, "c0")
    # # m.addConstr(y <= 2, "c0")

    m.optimize()

    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % m.objVal)

except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')
