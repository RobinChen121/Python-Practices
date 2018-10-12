# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 21:02:27 2018

@author: Zhen Chen

@description:  compute Ly 
    
"""

import scipy.stats as st

price = 4
holdCost = 1
meand1 = 6.6
meand2 = 2
variCost = 1
salValue = 0

I1 = 0    # expected inventory in the first period for an order up to level y
y = input("input an order-up-to level y: ")
y = int(y)
for i in range(y):
    I1 = (y - i ) * st.poisson.pmf(i, meand1) + I1
    
print("expected inventory is %.5f in the first period" % I1)
Icost = holdCost * I1
print("expected inventory cost is %.5f in the first period" % Icost)
Ly1 = (price - variCost) * y - holdCost * I1  + salValue * I1 
print("expected Ly1 is %.5f in the first period" % Ly1)

I12 = 0 # expected inventory in two periods for an order up to level y
for i in range(y):
    I12 = (y - i ) * st.poisson.pmf(i, meand1 + meand2) + I12
print("expected inventory I12 is %.5f in the second period" % I12)
Ly = (price - variCost) * y - holdCost * I12 - holdCost * I1 + salValue * I12 
print("expected Ly12 is %.5f in two periods" % Ly)

I2 = 0 # expected inventory in second period for an order up to level y in second period 
for i in range(y):
    I2 = (y - i ) * st.poisson.pmf(i, meand2) + I2
Ly2 = (price - variCost) * y - holdCost * I2  + salValue * I2 
print("expected Ly2 is %.5f in the second period" % Ly2)

