#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 20:04:23 2023

@author: zhenchen

@disp:  
    overdraft for single product situation
    
"""

ini_I = 0
ini_cash = 0
vari_cost = 1
price = 5
mean_demands = [10, 20]
overhead_cost = [50, 50]
r0 = 0.01
r0 = 0.1
limit = 20 # overdraft limit
T = len(mean_demands)


sample_nums = [10 for t in range(T)]


