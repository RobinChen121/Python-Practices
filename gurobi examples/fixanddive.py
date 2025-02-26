#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

# Implement a simple MIP heuristic.  Relax the model,
# sort variables based on fractionality, and fix the 25% of
# the fractional variables that are closest to integer variables.
# Repeat until either the relaxation is integer feasible or
# linearly infeasible.

import sys
import gurobipy as gp
from gurobipy import GRB


# Key function used to sort variables based on relaxation fractionality


def sortkey(v1):
    sol = v1.X
    return abs(sol - int(sol + 0.5))


if len(sys.argv) < 2:
    print("Usage: fixanddive.py filename")
    sys.exit(0)

# Read model

model = gp.read(sys.argv[1])

# Collect integer variables and relax them
intvars = []
for v in model.getVars():
    if v.VType != GRB.CONTINUOUS:
        intvars += [v]
        v.VType = GRB.CONTINUOUS

model.Params.OutputFlag = 0

model.optimize()


# Perform multiple iterations.  In each iteration, identify the first
# quartile of integer variables that are closest to an integer value in the
# relaxation, fix them to the nearest integer, and repeat.

for iter in range(1000):
    # create a list of fractional variables, sorted in order of increasing
    # distance from the relaxation solution to the nearest integer value

    fractional = []
    for v in intvars:
        sol = v.X
        if abs(sol - int(sol + 0.5)) > 1e-5:
            fractional += [v]

    fractional.sort(key=sortkey)

    print(f"Iteration {iter}, obj {model.ObjVal:g}, fractional {len(fractional)}")

    if len(fractional) == 0:
        print(f"Found feasible solution - objective {model.ObjVal:g}")
        break

    # Fix the first quartile to the nearest integer value
    nfix = max(int(len(fractional) / 4), 1)
    for i in range(nfix):
        v = fractional[i]
        fixval = int(v.X + 0.5)
        v.LB = fixval
        v.UB = fixval
        print(f"  Fix {v.VarName} to {fixval:g} (rel {v.X:g})")

    model.optimize()

    # Check optimization result

    if model.Status != GRB.OPTIMAL:
        print("Relaxation is infeasible")
        break
