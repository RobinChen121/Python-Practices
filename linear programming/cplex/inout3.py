#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: inout3.py
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
# inout3.py -- A production planning problem
#
# Problem Description
# -------------------
#
# A company has to produce 3 products, using 2 resources.
# Each resource has a limited capacity.
# Each product consumes a given number of machines.
# Each product has a production cost (the inside cost).
# Both products can also be purchased outside the company at a given
# cost (the outside cost).
#
# Minimize external production given product demand, a cost
# constraint, and minimum internal production constraints.
#
# To run from the command line, use
#
#    python inout3.py

from __future__ import print_function

import sys

import cplex
from cplex import SparsePair

consumption = [[0.5, 0.4, 0.3], [0.2, 0.4, 0.6]]
capacity = [20.0, 40.0]
demand = [100.0, 200.0, 300.0]
insideCost = [0.6, 0.8, 0.3]
outsideCost = [0.8, 0.9, 0.4]

nbProducts = len(demand)
nbResources = len(capacity)


def inout3():
    c = cplex.Cplex()

    # sys.stdout is the default output stream for log and results
    # so these lines may be omitted
    c.set_results_stream(sys.stdout)
    c.set_log_stream(sys.stdout)

    # indices of the inside production variables
    inside = list(range(0, nbProducts))
    c.variables.add(lb=[10.0 for x in inside],
                    names=["inside_" + str(i) for i in range(nbProducts)])

    # indices of the outside production variables
    outside = list(range(nbProducts, 2 * nbProducts))
    c.variables.add(lb=[0.0 for x in outside],
                    names=["outside_" + str(i) for i in range(nbProducts)])

    # index of the cost variables
    cost = 2 * nbProducts
    c.variables.add(obj=[1.0], names=["cost"])

    # assign the cost varibles
    c.linear_constraints.add(lin_expr=[SparsePair(ind=[cost] +
                                                  inside + outside,
                                                  val=[-1.0] +
                                                  insideCost + outsideCost)],
                             senses="E", rhs=[0.0], names=["cost"])

    # add capacity constraint for each resource
    c.linear_constraints.add(lin_expr=[SparsePair(ind=inside,
                                                  val=consumption[i])
                                       for i in range(len(consumption))],
                             senses=["L" for i in consumption],
                             rhs=capacity,
                             names=["capacity_" + str(i)
                                    for i in range(nbResources)])

    # must meet demand for each product
    c.linear_constraints.add(lin_expr=[SparsePair(ind=[inside[p]] +
                                                  [outside[p]],
                                                  val=[1.0 for i in [0, 1]])
                                       for p in range(nbProducts)],
                             senses=["E" for i in demand],
                             rhs=demand,
                             names=["demand_" + str(i)
                                    for i in range(nbProducts)])

    # find cost-minimal solution
    c.solve()
    print("Solution status = ", c.solution.get_status())

    # Add constraint: cost must be no more than 10% over minimum
    c.variables.set_upper_bounds(cost, 1.1 * c.solution.get_objective_value())

    # Set objective to minimize outside production
    c.objective.set_linear(cost, 0.0)
    c.objective.set_linear([outside[i], 1.0] for i in range(len(outside)))

    c.write("inout3.lp")

    # optimize for new objective
    c.solve()
    print("Solution status = ", c.solution.get_status())
    # display the solution
    print("cost: ", c.solution.get_values(cost))
    for p in range(nbProducts):
        print("Product ", p, ":")
        print("   inside: ", c.solution.get_values(inside[p]))
        print("  outside: ", c.solution.get_values(outside[p]))

if __name__ == "__main__":
    inout3()
