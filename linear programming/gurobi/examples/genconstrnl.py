import gurobipy as gp
from gurobipy import nlfunc


# Formulate and solve the simple nonlinear model

# Minimize y
# s.t.     y = sin(2.5 x1) + x2
#          -1 <= x1, x2 <= 1

with gp.Env() as env, gp.Model(env=env) as model:
    # Optimization variables
    x1 = model.addVar(lb=-1, ub=1, name="x1")
    x2 = model.addVar(lb=-1, ub=1, name="x2")

    # Auxiliary resultant variable for general constraint
    y = model.addVar(lb=-float("inf"), name="y")

    # Nonlinear constraint for y
    model.addGenConstrNL(y, nlfunc.sin(2.5 * x1) + x2)

    # Use y for objective function
    model.setObjective(y)

    model.optimize()

    print(f"x1={x1.X}  x2={x2.X}  obj={y.X}")
