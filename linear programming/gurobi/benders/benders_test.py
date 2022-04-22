# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 20:47:40 2021

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: use benders method for a numerical example:
    
  max  4x_1+7x_2+2y_1-3y_2+y_3
s.t.
       2x_1+4x_2+4y_1-2y_2+3y_3 <= 12
       3x_1+5x_2+2y_1+3y_2-y_3 <= 10
       x_1 <= 2, x_2 <= 2 
       y_1 >= 0, y_2 >= 0, y_3 >= 0
       x_1, x_2 为整数
    
"""

from gurobipy import *
import numpy as np
import math


# \dsp(x) of the second stage problem, where x is given 
def DSP(x):
    global b 
    global A
    global B
    global c
    global d
    
    n = len(b) # the number of decision varibales in the second stage
    
    try:
        m2 = Model("second-stage")
        u = m2.addMVar(n, vtype=GRB.CONTINUOUS) # gurobi currently support only for 1-D MVar objects, not 2-D
        
        # Set objective
        coe = b - A@x
        
        m2.setObjective(coe @ u, GRB.MAXIMIZE)
        m2.addConstr(B.T @ u <= d)
        
        m2.update()
        #m2.setParam('DualReductions', 0)
        m2.Params.InfUnbdInfo = 1  # or m2.setParam('InfUnbdInfo', 1), Determines whether simplex (and crossover) will compute additional information when a model
                                   # is determined to be infeasible or unbounded
        m2.optimize()
        
        if m2.Status == 5: # model is unbounded
            v = m2.unbdray
            return m2.Status, v # 若无界返回求解状态与基射线
        if m2.Status == 2: # model is optimal
            nita = m2.objVal
            u_value = [u.X[i] for i in range(n)]
            return m2.Status, u_value, nita  # 若有可行解，返回求解状态与可行值
            
        # if m2.Status == 3:
        #     print("Model is infeasible")
        #     m2.computeIIS()
        #     m2.write("model.ilp")
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))




b = np.array([-12, -10])
A = np.array([[-2, -4], [-3, -5]])
B = np.array([[-4, 2, -3], [-2, -3, 1]])
c = np.array([-4, -7])
d = np.array([-2, 3, -1])
DSP([1, 1])
    
try:     
    z_ub = GRB.INFINITY
    z_lb = -GRB.INFINITY
    
    # initialize a lower bound for second stage problem
    nita_lb = -200 
    
    n = len(c) # the number of decision varibales in the first stage
    m1 = Model("first-stage")
    x = m1.addMVar(n, ub = 2, vtype=GRB.CONTINUOUS) # gurobi currently support only for 1-D MVar objects, not 2-D
    nita = m1.addMVar(1, lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS)
    
    m1.addConstr(nita >= nita_lb)
    m1.setObjective(c @ x + nita, GRB.MINIMIZE)
    m1.optimize()
    
    x_value = [x.X[i] for i in range(n)]
    z_lb = m1.objVal
    
    last_nita_value = -GRB.INFINITY # nita 的一个初始值，若迭代时 nita 的值不发生变化，则开始分支定界
    all_integer = [False for i in range(n)] # 判断决策变量是不是整数型
    while abs(z_ub - z_lb) > 1e-6:
        result = DSP(x_value)
        if result[0] == 5: # 无界
            ray = result[1]
            m1.addConstr(ray@b - ray@A@x <= 0) # 添加 feasibility cut
            
            m1.update()
            m1.optimize()
            x_value = [x.X[i] for i in range(n)]
            z_lb = m1.objVal           
            print()
        if result[0] == 2: # 可行解
            u_value = result[1]
            nita_value = result[2]
            if abs(last_nita_value - nita_value) < 1e-6: # 开始分支定界
                for i in range(n):
                    if abs(x_value[i] - math.floor(x_value[i])) < 1e-6:
                        all_integer[i] = True
                    else:        
                        # 这里面的分支定界目前是不完善的，编程实现分支定界要用到递归，
                        # 即模型不断调用自己
                        m1.addConstr(x[i] <= math.floor(x_value[i]))
                        break
                if sum(all_integer) == n:
                    z_ub = c @ x_value + nita_value
                    if abs(z_ub - z_lb) < 1e-6:
                        print('Obj: %g' % m1.objVal)
                        print('x = ')
                        print(x_value)
                        break
                               
            m1.addConstr(b@u_value - u_value@A@x <= nita) # 添加 optimality cut

            m1.update()            
            m1.optimize()
            x_value = [x.X[i] for i in range(n)]
            z_lb = m1.objVal
            last_nita_value = nita.X[0]  
    
    
except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')

