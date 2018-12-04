#!/usr/bin/python

# Copyright 2018, Gurobi Optimization, LLC

# Solve a model with different values of the Method parameter;
# show which value gives the shortest solve time.

import sys
from gurobipy import *

if len(sys.argv) < 2:
    print('Usage: lpmethod.py filename')
    quit()

# Read model
m = read(sys.argv[1])

# Solve the model with different values of Method
bestTime = m.Params.timeLimit
bestMethod = -1
for i in range(3):
    m.reset()
    m.Params.method = i
    m.optimize()
    if m.status == GRB.Status.OPTIMAL:
        bestTime = m.Runtime
        bestMethod = i
        # Reduce the TimeLimit parameter to save time with other methods
        m.Params.timeLimit = bestTime

# Report which method was fastest
if bestMethod == -1:
    print('Unable to solve this model')
else:
    print('Solved in %g seconds with Method %d' % (bestTime, bestMethod))
