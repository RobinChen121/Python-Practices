#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:15:13 2023

@author: chen
"""

from gurobipy import *

pi = 6
h = 4
m = Model()
q = m.addVar(lb = 0, vtype = GRB.CONTINUOUS)
Iplus = m.addVar(lb = 0, vtype = GRB.CONTINUOUS)
Iminus = m.addVar(vtype = GRB.CONTINUOUS)

d = 3 
obj = h * Iplus + pi * Iminus
m.setObjective(obj, GRB.MINIMIZE)
m.addConstr(Iplus >= q - d)
m.addConstr(Iminus >= d - q)
m.addConstr(Iplus <= q)
m.addConstr(Iminus <= d)


m.optimize()
objV = m.objVal
qv = q.x
piV = m.getAttr(GRB.Attr.Pi)
