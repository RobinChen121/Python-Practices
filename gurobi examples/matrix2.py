#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

# This example uses the matrix friendly API to formulate the n-queens
# problem; it maximizes the number queens placed on an n x n
# chessboard without threatening each other.
#
# This example demonstrates slicing on MVar objects.

import numpy as np
import gurobipy as gp
from gurobipy import GRB

n = 8

m = gp.Model("nqueens")

# n-by-n binary variables; x[i, j] decides whether a queen is placed at
# position (i, j)
x = m.addMVar((n, n), vtype=GRB.BINARY, name="x")

# Maximize the number of placed queens
m.setObjective(x.sum(), GRB.MAXIMIZE)

# At most one queen per row; this adds n linear constraints
m.addConstr(x.sum(axis=1) <= 1, name="row")

# At most one queen per column; this adds n linear constraints
m.addConstr(x.sum(axis=0) <= 1, name="col")

for i in range(-n + 1, n):
    # At most one queen on diagonal i
    m.addConstr(x.diagonal(i).sum() <= 1, name=f"diag{i:d}")

    # At most one queen on anti-diagonal i
    m.addConstr(x[:, ::-1].diagonal(i).sum() <= 1, name=f"adiag{i:d}")

# Solve the problem
m.optimize()

print(x.X)
print(f"Queens placed: {m.ObjVal:.0f}")
