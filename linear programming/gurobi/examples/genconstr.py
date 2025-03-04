#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

# In this example we show the use of general constraints for modeling
# some common expressions. We use as an example a SAT-problem where we
# want to see if it is possible to satisfy at least four (or all) clauses
# of the logical form
#
# L = (x0 or ~x1 or x2)  and (x1 or ~x2 or x3)  and
#     (x2 or ~x3 or x0)  and (x3 or ~x0 or x1)  and
#     (~x0 or ~x1 or x2) and (~x1 or ~x2 or x3) and
#     (~x2 or ~x3 or x0) and (~x3 or ~x0 or x1)
#
# We do this by introducing two variables for each literal (itself and its
# negated value), one variable for each clause, one variable indicating
# whether we can satisfy at least four clauses, and one last variable to
# identify the minimum of the clauses (so if it is one, we can satisfy all
# clauses). Then we put these last two variables in the objective.
# The objective function is therefore
#
# maximize Obj0 + Obj1
#
#  Obj0 = MIN(Clause1, ... , Clause8)
#  Obj1 = 1 -> Clause1 + ... + Clause8 >= 4
#
# thus, the objective value will be two if and only if we can satisfy all
# clauses; one if and only if at least four but not all clauses can be satisfied,
# and zero otherwise.

import gurobipy as gp
from gurobipy import GRB
import sys

try:
    NLITERALS = 4

    n = NLITERALS

    # Example data:
    #   e.g. {0, n+1, 2} means clause (x0 or ~x1 or x2)
    Clauses = [
        [0, n + 1, 2],
        [1, n + 2, 3],
        [2, n + 3, 0],
        [3, n + 0, 1],
        [n + 0, n + 1, 2],
        [n + 1, n + 2, 3],
        [n + 2, n + 3, 0],
        [n + 3, n + 0, 1],
    ]

    # Create a new model
    model = gp.Model("Genconstr")

    # initialize decision variables and objective
    Lit = model.addVars(NLITERALS, vtype=GRB.BINARY, name="X")
    NotLit = model.addVars(NLITERALS, vtype=GRB.BINARY, name="NotX")

    Cla = model.addVars(len(Clauses), vtype=GRB.BINARY, name="Clause")

    Obj0 = model.addVar(vtype=GRB.BINARY, name="Obj0")
    Obj1 = model.addVar(vtype=GRB.BINARY, name="Obj1")

    # Link Xi and notXi
    model.addConstrs(
        (Lit[i] + NotLit[i] == 1.0 for i in range(NLITERALS)), name="CNSTR_X"
    )

    # Link clauses and literals
    for i, c in enumerate(Clauses):
        clause = []
        for l in c:
            if l >= n:
                clause.append(NotLit[l - n])
            else:
                clause.append(Lit[l])
        model.addConstr(Cla[i] == gp.or_(clause), "CNSTR_Clause" + str(i))

    # Link objs with clauses
    model.addConstr(Obj0 == gp.min_(Cla.values()), name="CNSTR_Obj0")
    model.addConstr((Obj1 == 1) >> (Cla.sum() >= 4.0), name="CNSTR_Obj1")

    # Set optimization objective
    model.setObjective(Obj0 + Obj1, GRB.MAXIMIZE)

    # Save problem
    model.write("genconstr.mps")
    model.write("genconstr.lp")

    # Optimize
    model.optimize()

    # Status checking
    status = model.getAttr(GRB.Attr.Status)

    if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        print("The model cannot be solved because it is infeasible or unbounded")
        sys.exit(1)

    if status != GRB.OPTIMAL:
        print("Optimization was stopped with status ", status)
        sys.exit(1)

    # Print result
    objval = model.getAttr(GRB.Attr.ObjVal)

    if objval > 1.9:
        print("Logical expression is satisfiable")
    elif objval > 0.9:
        print("At least four clauses can be satisfied")
    else:
        print("Not even three clauses can be satisfied")

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")
