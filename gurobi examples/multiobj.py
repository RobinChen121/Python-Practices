#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

# Want to cover four different sets but subject to a common budget of
# elements allowed to be used. However, the sets have different priorities to
# be covered; and we tackle this by using multi-objective optimization.

import gurobipy as gp
from gurobipy import GRB
import sys

try:
    # Sample data
    Groundset = range(20)
    Subsets = range(4)
    Budget = 12
    Set = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    ]
    SetObjPriority = [3, 2, 2, 1]
    SetObjWeight = [1.0, 0.25, 1.25, 1.0]

    # Create initial model
    model = gp.Model("multiobj")

    # Initialize decision variables for ground set:
    # x[e] == 1 if element e is chosen for the covering.
    Elem = model.addVars(Groundset, vtype=GRB.BINARY, name="El")

    # Constraint: limit total number of elements to be picked to be at most
    # Budget
    model.addConstr(Elem.sum() <= Budget, name="Budget")

    # Set global sense for ALL objectives
    model.ModelSense = GRB.MAXIMIZE

    # Limit how many solutions to collect
    model.setParam(GRB.Param.PoolSolutions, 100)

    # Set and configure i-th objective
    for i in Subsets:
        objn = sum(Elem[k] * Set[i][k] for k in range(len(Elem)))
        model.setObjectiveN(
            objn, i, SetObjPriority[i], SetObjWeight[i], 1.0 + i, 0.01, "Set" + str(i)
        )

    # Save problem
    model.write("multiobj.lp")

    # Optimize
    model.optimize()

    model.setParam(GRB.Param.OutputFlag, 0)

    # Status checking
    status = model.Status
    if status in (GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED):
        print("The model cannot be solved because it is infeasible or unbounded")
        sys.exit(1)

    if status != GRB.OPTIMAL:
        print(f"Optimization was stopped with status {status}")
        sys.exit(1)

    # Print best selected set
    print("Selected elements in best solution:")
    selected = [e for e in Groundset if Elem[e].X > 0.9]
    print(" ".join(f"El{e}" for e in selected))

    # Print number of solutions stored
    nSolutions = model.SolCount
    print(f"Number of solutions found: {nSolutions}")

    # Print objective values of solutions
    if nSolutions > 10:
        nSolutions = 10
    print(f"Objective values for first {nSolutions} solutions:")
    for i in Subsets:
        model.setParam(GRB.Param.ObjNumber, i)
        objvals = []
        for e in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber, e)
            objvals.append(model.ObjNVal)

        print(f"\tSet{i}" + "".join(f" {objval:6g}" for objval in objvals[:3]))

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError as e:
    print(f"Encountered an attribute error: {e}")
