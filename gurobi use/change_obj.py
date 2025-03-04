"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/3/4 18:13
Description: getObjective() can get objective function only after optimize().
    

"""

import gurobipy as gp
from gurobipy import GRB

try:
    # Create a new model
    m = gp.Model("mip1")

    # Create variables
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")



    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

    # Add constraint: x + y >= 1
    m.addConstr(x + y >= 1, "c1")

    # Set objective
    m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)


    # Optimize model
    m.optimize()
    obj = m.getObjective()  # can get objective function only after optimize
    print(obj)
    m.write('mip1.lp')

    for v in m.getVars():
        print(f"{v.VarName} {v.X:g}")

    print(f"Obj: {m.ObjVal:g}")

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")

except AttributeError:
    print("Encountered an attribute error")
