"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/20 11:46
Description: 
    The gurobi Model() object can't be pickled.

"""

import multiprocessing
import numpy as np
from gurobipy import Model, GRB, Env
import time


# Create a new model
m = Model()
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


# noinspection PyTypeChecker
def solve_pool(rhs_):
    m.update()
    m.setAttr("RHS", m.getConstrs()[0], rhs_)
    m.optimize()
    return m.ObjVal


def solve_process(queue_, rhs_):
    m.update()
    m.setAttr("RHS", m.getConstrs()[0], rhs_)
    m.optimize()
    queue_.put(m.ObjVal)
    # print(rhs_)
    # queue_.put(rhs_)


def solve_lock(queue_, lock_, rhs_):

    m.update()
    m.setAttr("RHS", m.getConstrs()[0], rhs_)
    m.optimize()
    with lock_:
        queue_.put(m.ObjVal)


if __name__ == "__main__":
    N = 10
    rhs = np.arange(1, N + 1)

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

    # multiprocessing.set_start_method("forkserver")  #("spawn")  # 或 "forkserver"
    q = multiprocessing.Queue()
    processes = []
    for i in range(10): # the range can affect multiprocessing, very weired
        p = multiprocessing.Process(target=solve_process, args=(q, 5))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    result = [q.get() for _ in range(10)]
    print(result)

    time_start = time.time()
    results = []
    for item in rhs:
        solve_pool(item)
        results.append(solve_pool(item))
    time_end = time.time()
    print(f"Time cost for sequential :{time_end - time_start:.4f}s\n" )
    # print(results)

    time_start = time.time()
    with multiprocessing.Pool() as pool:
        result = pool.map(solve_pool, rhs)
    time_end = time.time()
    print(f"Time cost for parallel :{time_end - time_start:.4f}s\n")
    # print(result)
