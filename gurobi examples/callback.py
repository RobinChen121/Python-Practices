#!/usr/bin/env python3.11

# Copyright 2025, Gurobi Optimization, LLC

#   This example reads a model from a file, sets up a callback that
#   monitors optimization progress and implements a custom
#   termination strategy, and outputs progress information to the
#   screen and to a log file.
#
#   The termination strategy implemented in this callback stops the
#   optimization of a MIP model once at least one of the following two
#   conditions have been satisfied:
#     1) The optimality gap is less than 10%
#     2) At least 10000 nodes have been explored, and an integer feasible
#        solution has been found.
#   Note that termination is normally handled through Gurobi parameters
#   (MIPGap, NodeLimit, etc.).  You should only use a callback for
#   termination if the available parameters don't capture your desired
#   termination criterion.

import sys
from functools import partial

import gurobipy as gp
from gurobipy import GRB


class CallbackData:
    def __init__(self, modelvars):
        self.modelvars = modelvars
        self.lastiter = -GRB.INFINITY
        self.lastnode = -GRB.INFINITY


def mycallback(model, where, *, cbdata, logfile):
    """
    Callback function. 'model' and 'where' arguments are passed by gurobipy
    when the callback is invoked. The other arguments must be provided via
    functools.partial:
      1) 'cbdata' is an instance of CallbackData, which holds the model
         variables and tracks state information across calls to the callback.
      2) 'logfile' is a writeable file handle.
    """

    if where == GRB.Callback.POLLING:
        # Ignore polling callback
        pass
    elif where == GRB.Callback.PRESOLVE:
        # Presolve callback
        cdels = model.cbGet(GRB.Callback.PRE_COLDEL)
        rdels = model.cbGet(GRB.Callback.PRE_ROWDEL)
        if cdels or rdels:
            print(f"{cdels} columns and {rdels} rows are removed")
    elif where == GRB.Callback.SIMPLEX:
        # Simplex callback
        itcnt = model.cbGet(GRB.Callback.SPX_ITRCNT)
        if itcnt - cbdata.lastiter >= 100:
            cbdata.lastiter = itcnt
            obj = model.cbGet(GRB.Callback.SPX_OBJVAL)
            ispert = model.cbGet(GRB.Callback.SPX_ISPERT)
            pinf = model.cbGet(GRB.Callback.SPX_PRIMINF)
            dinf = model.cbGet(GRB.Callback.SPX_DUALINF)
            if ispert == 0:
                ch = " "
            elif ispert == 1:
                ch = "S"
            else:
                ch = "P"
            print(f"{int(itcnt)} {obj:g}{ch} {pinf:g} {dinf:g}")
    elif where == GRB.Callback.MIP:
        # General MIP callback
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
        objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
        solcnt = model.cbGet(GRB.Callback.MIP_SOLCNT)
        if nodecnt - cbdata.lastnode >= 100:
            cbdata.lastnode = nodecnt
            actnodes = model.cbGet(GRB.Callback.MIP_NODLFT)
            itcnt = model.cbGet(GRB.Callback.MIP_ITRCNT)
            cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
            print(
                f"{nodecnt:.0f} {actnodes:.0f} {itcnt:.0f} {objbst:g} "
                f"{objbnd:g} {solcnt} {cutcnt}"
            )
        if abs(objbst - objbnd) < 0.1 * (1.0 + abs(objbst)):
            print("Stop early - 10% gap achieved")
            model.terminate()
        if nodecnt >= 10000 and solcnt:
            print("Stop early - 10000 nodes explored")
            model.terminate()
    elif where == GRB.Callback.MIPSOL:
        # MIP solution callback
        nodecnt = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        solcnt = model.cbGet(GRB.Callback.MIPSOL_SOLCNT)
        x = model.cbGetSolution(cbdata.modelvars)
        print(
            f"**** New solution at node {nodecnt:.0f}, obj {obj:g}, "
            f"sol {solcnt:.0f}, x[0] = {x[0]:g} ****"
        )
    elif where == GRB.Callback.MIPNODE:
        # MIP node callback
        print("**** New node ****")
        if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            x = model.cbGetNodeRel(cbdata.modelvars)
            model.cbSetSolution(cbdata.modelvars, x)
    elif where == GRB.Callback.BARRIER:
        # Barrier callback
        itcnt = model.cbGet(GRB.Callback.BARRIER_ITRCNT)
        primobj = model.cbGet(GRB.Callback.BARRIER_PRIMOBJ)
        dualobj = model.cbGet(GRB.Callback.BARRIER_DUALOBJ)
        priminf = model.cbGet(GRB.Callback.BARRIER_PRIMINF)
        dualinf = model.cbGet(GRB.Callback.BARRIER_DUALINF)
        cmpl = model.cbGet(GRB.Callback.BARRIER_COMPL)
        print(f"{itcnt:.0f} {primobj:g} {dualobj:g} {priminf:g} {dualinf:g} {cmpl:g}")
    elif where == GRB.Callback.MESSAGE:
        # Message callback
        msg = model.cbGet(GRB.Callback.MSG_STRING)
        logfile.write(msg)


# Parse arguments
if len(sys.argv) < 2:
    print("Usage: callback.py filename")
    sys.exit(0)
model_file = sys.argv[1]

# This context block manages several resources to ensure they are properly
# closed at the end of the program:
#   1) A Gurobi environment, with console output and heuristics disabled.
#   2) A Gurobi model, read from a file provided by the user.
#   3) A Python file handle which the callback will write to.

with gp.Env(params={"OutputFlag": 0, "Heuristics": 0}) as env, gp.read(
    model_file, env=env
) as model, open("cb.log", "w") as logfile:

    # Set up callback function with required arguments
    callback_data = CallbackData(model.getVars())
    callback_func = partial(mycallback, cbdata=callback_data, logfile=logfile)

    # Solve model and print solution information
    model.optimize(callback_func)

    print("")
    print("Optimization complete")
    if model.SolCount == 0:
        print(f"No solution found, optimization status = {model.Status}")
    else:
        print(f"Solution found, objective = {model.ObjVal:g}")
        for v in model.getVars():
            if v.X != 0.0:
                print(f"{v.VarName} {v.X:g}")
