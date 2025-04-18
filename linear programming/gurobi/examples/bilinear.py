#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

# This example formulates and solves the following simple bilinear model:
#  maximize    x
#  subject to  x + y + z <= 10
#              x * y <= 2         (bilinear inequality)
#              x * z + y * z = 1  (bilinear equality)
#              x, y, z non-negative (x integral in second version)

import gurobipy as gp
from gurobipy import GRB

# Create a new model
m = gp.Model("bilinear")

# Create variables
x = m.addVar(name="x")
y = m.addVar(name="y")
z = m.addVar(name="z")

# Set objective: maximize x
m.setObjective(1.0 * x, GRB.MAXIMIZE)

# Add linear constraint: x + y + z <= 10
m.addConstr(x + y + z <= 10, "c0")

# Add bilinear inequality constraint: x * y <= 2
m.addConstr(x * y <= 2, "bilinear0")

# Add bilinear equality constraint: x * z + y * z == 1
m.addConstr(x * z + y * z == 1, "bilinear1")

# Optimize model
m.optimize()

m.printAttr("x")

# Constrain 'x' to be integral and solve again
x.VType = GRB.INTEGER
m.optimize()

m.printAttr("x")
