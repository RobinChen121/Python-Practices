""" 
# @File  : mip_deterministic_noFinance.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2019/11/6
# @Desc  : the mip model for no finance situation when demand are deterministic

"""

import numpy as np
from gurobipy import *
#from gurobipy import Model
#from gurobipy import GurobiError

# parameter values
ini_cash = 10000
ini_I1 = 0
ini_I2 = 0

price1 = 50
price2 = 20
vari_cost1 = 25
vari_cost2 = 10
overhead_cost = 500

T = 24
delay_length = 2
discount_rate = 0.01

demand1 = 30 * np.ones((1, T))
demand2 = 50 * np.ones((1, T))


try:
    # Create a new model
    m = Model("self-cash-mip")

    # Create variables
    Q1 = {} # ordering quantity in each period for product 1
    Q2 = {} # ordering quantity in each period for product 2
    I1 = {} # end-of-period inventory in each period for product 1
    I2 = {} # end-of-period inventory in each period for product 2
    C = {} # end-of-period cash in each period
    for i in range(T):
        Q1[i] = m.addVar(vtype=GRB.CONTINUOUS)
        Q2[i] = m.addVar(vtype=GRB.CONTINUOUS)
        I1[i] = m.addVar(vtype=GRB.CONTINUOUS)
        I2[i] = m.addVar(vtype=GRB.CONTINUOUS)
        C[i] = m.addVar(vtype=GRB.CONTINUOUS)

    # objective function




except GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))
    
except AttributeError:
    print('Encountered an attribute error')

