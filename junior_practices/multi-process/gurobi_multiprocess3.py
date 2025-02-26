"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/26 12:38
Description: 
    gurobi/cplex model can't be pickled.

"""
import multiprocessing
import numpy as np
from gurobipy import Model, GRB, Env
import time
import pickle


# noinspection PyTypeChecker
def solve_pool(rhs_):
    with Env() as env_, Model(env=env_) as model:
        x_ = model.addVar(
            vtype=GRB.BINARY, name="x"
        )  # default bounds for continuous type is [0, infinite]
        y_ = model.addVar(vtype=GRB.BINARY, name="y")
        z_ = model.addVar(vtype=GRB.BINARY, name="z")

        # Set objective
        model.setObjective(x_ + y_ + 2 * z_, GRB.MAXIMIZE)

        # Add constraint: x + 2 y + 3 z <= 4
        model.addConstr(x_ + 2 * y_ + 3 * z_ <= 4, "c0")

        # Add constraint: x + y >= 1
        model.addConstr(x_ + y_ >= 1, "c1")

        model.update()
        model.setAttr("RHS", model.getConstrs()[0], rhs_)
        model.optimize()
        return model.ObjVal

if __name__ == "__main__":
    # Create a new model
    env = Env(params={"OutputFlag": 0})
    m = Model(env=env)
    m.params.OutputFlag = 0

    # Create variables
    x = m.addVar(
        vtype=GRB.BINARY, name="x"
    )  # default bounds for continuous type is [0, infinite]
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")

    # Set objective
    m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

    # Add constraint: x + y >= 1
    m.addConstr(x + y >= 1, "c1")

    pickle_model = pickle.dumps(m)

    # rhs = np.arange(1, 10)
    # # works well using pool
    # time_start = time.time()
    # with multiprocessing.Pool() as pool:
    #     result = pool.map(solve_pool, rhs)
    # time_end = time.time()
    # print(f"Time cost for parallel :{time_end - time_start:.4f}s\n")
    # print(result)