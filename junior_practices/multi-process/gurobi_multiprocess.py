"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/20 11:46
Description: 
    

"""
import multiprocessing
import numpy as np
from gurobipy import Model, GRB, Env


def solve(rhs_):
    # Create a new model

    m = Model()
    m.params.OutputFlag = 0

    # Create variables
    x = m.addVar(vtype=GRB.BINARY, name="x")  # default bounds for continuous type is [0, infinite]
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")

    # Set objective
    m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

    # Add constraint: x + y >= 1
    m.addConstr(x + y >= 1, "c1")

    m.update()
    m.setAttr("RHS", m.getConstrs()[0], rhs_)

    m.optimize()
    return m.objVal

if __name__ == "__main__":
    # processes = []
    # for i in range(2):
    #     p = multiprocessing.Process(target=solve)
    #     processes.append(p)
    #     p.start()
    #
    # for p in processes:
    #     p.join()

    rhs = np.arange(1, 10)
    result = solve(4)
    print(result)

    with multiprocessing.Pool() as pool:
        result = pool.map(solve, rhs)
        # result = pool.starmap(solve, [() for i in range(10)])
    print(result)
