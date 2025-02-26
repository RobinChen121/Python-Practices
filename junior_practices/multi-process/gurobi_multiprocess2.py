"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/25 18:24
Description: 
    

No issue happens when building Gurobi model in the target function of parallel.

"""

import multiprocessing
import numpy as np
from gurobipy import Model, GRB, Env
import time


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


def solve_process(queue_, rhs_):
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
        queue_.put(model.ObjVal)


if __name__ == "__main__":
    # # Create a new model
    # env = Env(params={"OutputFlag": 0})
    # m = Model(env=env)
    # m.params.OutputFlag = 0
    #
    # # Create variables
    # x = m.addVar(
    #     vtype=GRB.BINARY, name="x"
    # )  # default bounds for continuous type is [0, infinite]
    # y = m.addVar(vtype=GRB.BINARY, name="y")
    # z = m.addVar(vtype=GRB.BINARY, name="z")
    #
    # # Set objective
    # m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
    #
    # # Add constraint: x + 2 y + 3 z <= 4
    # m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
    #
    # # Add constraint: x + y >= 1
    # m.addConstr(x + y >= 1, "c1")

    # rhs = np.arange(1, 10)
    # # works well using pool
    # time_start = time.time()
    # with multiprocessing.Pool() as pool:
    #     result = pool.map(solve_pool, rhs)
    # time_end = time.time()
    # print(f"Time cost for parallel :{time_end - time_start:.4f}s\n")
    # print(result)

    # using Process has some issue
    q = multiprocessing.Queue()
    processes = []
    for i in range(1, 10):
        # if fixing i to some value and range() has only one arg, works well
        p = multiprocessing.Process(target=solve_process, args=(q, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    result = [q.get() for _ in range(1, 10)]
    print(result)

    # lock = multiprocessing.Lock()
    # queue = multiprocessing.Queue()
    # processes = []
    # for i in range(10):
    #     p = multiprocessing.Process(target=solve_lock, args=(queue, lock, i))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()
    # for i in range(10):
    #     print(queue.get())
