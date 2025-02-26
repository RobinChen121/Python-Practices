#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

# This example considers the following nonconvex nonlinear problem
#
#  minimize   sin(x) + cos(2*x) + 1
#  subject to  0.25*exp(x) - x <= 0
#              -1 <= x <= 4
#
#  We show you two approaches to solve it as a nonlinear model:
#
#  1) Set the parameter FuncNonlinear = 1 to handle all general function
#     constraints as true nonlinear functions.
#
#  2) Set the attribute FuncNonlinear = 1 for each general function
#     constraint to handle these as true nonlinear functions.
#

import gurobipy as gp
from gurobipy import GRB


def printsol(m, x):
    print(f"x = {x.X}")
    print(f"Obj = {m.ObjVal}")


try:
    # Create a new model
    m = gp.Model()

    # Create variables
    x = m.addVar(lb=-1, ub=4, name="x")
    twox = m.addVar(lb=-2, ub=8, name="2x")
    sinx = m.addVar(lb=-1, ub=1, name="sinx")
    cos2x = m.addVar(lb=-1, ub=1, name="cos2x")
    expx = m.addVar(name="expx")

    # Set objective
    m.setObjective(sinx + cos2x + 1, GRB.MINIMIZE)

    # Add linear constraints
    lc1 = m.addConstr(0.25 * expx - x <= 0)
    lc2 = m.addConstr(2.0 * x - twox == 0)

    # Add general function constraints
    # sinx = sin(x)
    gc1 = m.addGenConstrSin(x, sinx, "gc1")
    # cos2x = cos(twox)
    gc2 = m.addGenConstrCos(twox, cos2x, "gc2")
    # expx = exp(x)
    gc3 = m.addGenConstrExp(x, expx, "gc3")

    # Approach 1) Set FuncNonlinear parameter

    m.params.FuncNonlinear = 1

    # Optimize the model
    m.optimize()

    printsol(m, x)

    # Restore unsolved state and set parameter FuncNonlinear to
    # its default value
    m.reset()
    m.resetParams()

    # Approach 2) Set FuncNonlinear attribute for every
    #             general function constraint

    gc1.FuncNonlinear = 1
    gc2.FuncNonlinear = 1
    gc3.FuncNonlinear = 1

    m.optimize()

    printsol(m, x)

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")
