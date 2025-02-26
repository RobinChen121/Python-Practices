#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

# This example reads a MIP model from a file, solves it and prints
# the objective values from all feasible solutions generated while
# solving the MIP. Then it creates the associated fixed model and
# solves that model.

import sys
import gurobipy as gp
from gurobipy import GRB

if len(sys.argv) < 2:
    print("Usage: mip2.py filename")
    sys.exit(0)

# Read and solve model

model = gp.read(sys.argv[1])

if model.IsMIP == 0:
    print("Model is not a MIP")
    sys.exit(0)

model.optimize()

if model.Status == GRB.OPTIMAL:
    print(f"Optimal objective: {model.ObjVal:g}")
elif model.Status == GRB.INF_OR_UNBD:
    print("Model is infeasible or unbounded")
    sys.exit(0)
elif model.Status == GRB.INFEASIBLE:
    print("Model is infeasible")
    sys.exit(0)
elif model.Status == GRB.UNBOUNDED:
    print("Model is unbounded")
    sys.exit(0)
else:
    print(f"Optimization ended with status {model.Status}")
    sys.exit(0)

# Iterate over the solutions and compute the objectives
model.Params.OutputFlag = 0
print("")
for k in range(model.SolCount):
    model.Params.SolutionNumber = k
    print(f"Solution {k} has objective {model.PoolObjVal:g}")
print("")
model.Params.OutputFlag = 1

fixed = model.fixed()
fixed.Params.Presolve = 0
fixed.optimize()

if fixed.Status != GRB.OPTIMAL:
    print("Error: fixed model isn't optimal")
    sys.exit(1)

diff = model.ObjVal - fixed.ObjVal

if abs(diff) > 1e-6 * (1.0 + abs(model.ObjVal)):
    print("Error: objective values are different")
    sys.exit(1)

# Print values of nonzero variables
for v in fixed.getVars():
    if v.X != 0:
        print(f"{v.VarName} {v.X:g}")
