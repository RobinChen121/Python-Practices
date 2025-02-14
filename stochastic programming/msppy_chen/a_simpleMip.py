"""
Created on 2025/2/3, 21:15 

@author: Zhen Chen.

@Python version: 3.10

@description:

from introduction to stochastic programming John R. Birge, page 308:

min -2.5x_1 - 2x_2 + E{min{4.4y_1 + 3y_2}}
s.t.
   4x_1+5x_2 <= 15,
   x_1+x_2 >= 1.5,
   2y_1+3y_2 >= h_1+ \chi_1,
   4y_1+y_2 >= h_2 + \chi_2,
   \chi_1 = x_1 + 2x_2,
   \chi_2 = 2x_1 + x_2,
   y >= 0, integer

h^T = (-2.8, -1.2) and (-2, -3) with equal probability 0.5.
The original solution(-1.9) in the book may be wrong.

Using Extensive solver verifies the optimal value is -2.8. Thus, binarization
precision being 1 reach the same solution, so it is enough in this problem.

"""

from msm import MSIP
from solver_detail import Extensive,SDDiP
from msppy.evaluation import EvaluationTrue
import gurobipy
import numpy


precision = 1
numpy.random.seed(2)
MIP = MSIP(T = 2, bound = -10)
for t in range(2):
    m = MIP.models[t]
    # chi_past are local copies(real state variables for stage 2)
    # 2 chi_now and 2 chi_past
    # actually there are 4 chi variables in the model
    chi_now, chi_past = m.addStateVars(2, name = 'chi', lb = [1.5, 1.5], ub = [6, 7.5])
    if t == 0:
        x = m.addVars(2, obj = [-2.5, -2.0], name = 'x', lb = -gurobipy.GRB.INFINITY)
        slack = m.addVars(2, ub = 1/(10**precision))
        m.addConstr(4*x[0] + 5*x[1] <= 15)
        m.addConstr(x[0] + x[1] >= 1.5)
        m.addConstr(chi_now[0] == x[0] + 2*x[1] + slack[0])
        m.addConstr(chi_now[1] == 2*x[0] + x[1] + slack[1])
    else:
        y = m.addVars(2, obj = [4.4, 3.0], vtype = 'I', name = 'y')
        m.addConstr(
            2*y[0] + 3*y[1] - chi_past[0] >= 0, # the algorithm will make values chi_now = chi_past
            uncertainty = {'rhs': [-2.8, -2]}
        )
        m.addConstr(
            4*y[0] + 1*y[1] - chi_past[1] >= 0,
            uncertainty = {'rhs': [-1.2, -3]}
        )
print('extensive solver: ', Extensive(MIP).solve(outputFlag = 0))
MIP.binarize(bin_stage = 2, precision = precision)
SDDiP(MIP).solve(cuts = ['LG'], max_iterations = 128) # LG means Lagrange
# resultTrue = EvaluationTrue(MIP) # chen: there are some errors running multiprocessing simulation
# resultTrue.run(n_simulations=100)
# resultTrue.CI

####################################verification##############################################
### extensive model ##
#from gurobipy import *
#m = Model()
#y = m.addVars(2, 2, vtype = GRB.INTEGER, obj = [2.2, 1.5, 2.2, 1.5])
#x = m.addVars(2, obj = [-2.5, -2.0], name = 'x', lb = - GRB.INFINITY)
#xi = m.addVars(2, name = 'xi', lb = [1.4, 1.4], ub = [6, 7.5])
#m.update()
#m.addConstr(4 * x[0] + 5 * x[1] <= 15)
#m.addConstr(x[0] + x[1] >= 1.5)
#m.addConstr(xi[0] == x[0] + 2 * x[1])
#m.addConstr(xi[1] == x[1] + 2 * x[0])
#m.addConstr(2 * y[(0,0)] + 3 * y[(0,1)] - xi[0] >= -2.8)
#m.addConstr(4 * y[(0,0)] + 1 * y[(0,1)] - xi[1] >= -1.2)
#m.addConstr(2 * y[(1,0)] + 3 * y[(1,1)] - xi[0] >= -2.0)
#m.addConstr(4 * y[(1,0)] + 1 * y[(1,1)] - xi[1] >= -3.0)
#m.optimize()
####################################verification##############################################
