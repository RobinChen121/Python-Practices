#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 18:36:56 2024

@author: zhenchen

@disp:  
    test whether some subproblems have same dual values for constraits and similarities for original solution values
    
    
"""

from gurobipy import *



m_forward = Model()
q_forward = m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'q') 
q_pre_forward = m_forward.addVar(vtype = GRB.CONTINUOUS, name = 'qpre')
I_forward = m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'I')
cash_forward = m_forward[t][n].addVar(lb = -GRB.INFINITY, vtype = GRB.CONTINUOUS, name = 'C')
W0_forward = m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W0')
W1_forward = m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W1')
W2_forward = m_forward[t][n].addVar(vtype = GRB.CONTINUOUS, name = 'W2')