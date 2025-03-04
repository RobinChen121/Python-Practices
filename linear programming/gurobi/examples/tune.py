#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

#  This example reads a model from a file and tunes it.
#  It then writes the best parameter settings to a file
#  and solves the model using these parameters.

import sys
import gurobipy as gp

if len(sys.argv) < 2:
    print("Usage: tune.py filename")
    sys.exit(0)

# Read the model
model = gp.read(sys.argv[1])

# Set the TuneResults parameter to 2
#
# The first parameter setting is the result for the first solved
# setting. The second entry the parameter setting of the best parameter
# setting.
model.Params.TuneResults = 2

# Tune the model
model.tune()

if model.TuneResultCount >= 2:
    # Load the best tuned parameters into the model
    #
    # Note, the first parameter setting is associated to the first solved
    # setting and the second parameter setting to best tune result.
    model.getTuneResult(1)

    # Write tuned parameters to a file
    model.write("tune.prm")

    # Solve the model using the tuned parameters
    model.optimize()
