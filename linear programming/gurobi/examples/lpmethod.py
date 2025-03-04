#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

# Solve a model with different values of the Method parameter;
# show which value gives the shortest solve time.

import sys
import gurobipy as gp
from gurobipy import GRB

if len(sys.argv) < 2:
    print("Usage: lpmethod.py filename")
    sys.exit(0)

# Read model
m = gp.read(sys.argv[1])

# Solve the model with different values of Method
bestTime = m.Params.TimeLimit
bestMethod = -1
for i in range(3):
    m.reset()
    m.Params.Method = i
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        bestTime = m.Runtime
        bestMethod = i
        # Reduce the TimeLimit parameter to save time with other methods
        m.Params.TimeLimit = bestTime

# Report which method was fastest
if bestMethod == -1:
    print("Unable to solve this model")
else:
    print(f"Solved in {bestTime:g} seconds with Method {bestMethod}")
