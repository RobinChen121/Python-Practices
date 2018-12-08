#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: mipex2.py
# Version 12.7.0
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2016. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
#
# mipex2.py - Reading and optimizing a MIP problem.
#
# The user has to specify the file to read on the command line:
#
#    python mipex2.py <filename>

from __future__ import print_function

import sys

import cplex
from cplex.exceptions import CplexSolverError


def mipex2(filename):
    c = cplex.Cplex(filename)

    try:
        c.solve()
    except CplexSolverError:
        print("Exception raised during solve")
        return

    # solution.get_status() returns an integer code
    status = c.solution.get_status()
    print(c.solution.status[status])
    if status == c.solution.status.unbounded:
        print("Model is unbounded")
        return
    if status == c.solution.status.infeasible:
        print("Model is infeasible")
        return
    if status == c.solution.status.infeasible_or_unbounded:
        print("Model is infeasible or unbounded")
        return

    s_method = c.solution.get_method()
    s_type = c.solution.get_solution_type()

    print("Solution status = ", status, ":", end=' ')
    # the following line prints the status as a string
    print(c.solution.status[status])

    if s_type == c.solution.type.none:
        print("No solution available")
        return
    print("Objective value = ", c.solution.get_objective_value())

    print()

    x = c.solution.get_values(0, c.variables.get_num() - 1)
    # because we're querying the entire solution vector,
    # x = c.solution.get_values()
    # would have the same effect
    for j in range(c.variables.get_num()):
        print("Column %d: Value = %17.10g" % (j, x[j]))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: mipex2.py filename")
        print("  filename   Name of a file, with .mps, .lp, or .sav")
        print("             extension, and a possible, additional .gz")
        print("             extension")
        sys.exit(-1)
    mipex2(sys.argv[1])
