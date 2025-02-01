#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:33:30 2025

@author: zhen chen

@Python version: 3.10

@disp:  
    
    the air conditioner problem;

The data process is stage-wise independent and on the RHS.
It was originally from http://www.optimization-online.org/DB_FILE/2017/12/6388.pdf.
Verified optimal value by extensive solver is 68200.
SDDP solver also obtains the same optimal value.
d = (
    100 w.p. 0.4,
    300 w.p. 0.6
)
x is production quantity, y is inventory and w is overtime production quantity
The first stage:
min x + 3w + 0.5y
     x <= 2
     x + w - y = 1

The second stage:
min x + 3w + 0.5y
     x <= 2
     x + y_past + w - y_now = d

The third stage:

min x + 3w + 0.5y
     x <= 2
     x + y_past + w - y_now = d
     
"""

from msm import MSLP
from solver_detail import Extensive


T = 3
D = [100, 300]

airConditioner = MSLP(T, bound = 0)
for t in range(T):
    m = airConditioner[t]
    y_now, y_past = m.addStateVar(obj = 50, vtype =  'I', name = 'y')
    x = m.addVar(ub = 200, obj = 100, vtype = 'I', name = 'x')
    w = m.addVar(obj = 300, vtype = 'I', name = 'w')
    m.update()

    if t == 0:
        m.addConstr(x + w - y_now == 100)
    else:
        m.addConstr(y_past + x + w - y_now == 0,
            uncertainty = {'rhs': D})
        m.set_probability([0.4, 0.6])
    # m.write('air_me_' + str(t) + '.lp')

_, _, extensive_obj = Extensive(airConditioner).solve()
print("the obj of the extensive model is %.2f" % extensive_obj)
# SDDiP(airConditioner).solve(cuts =['B'], max_iterations=10)