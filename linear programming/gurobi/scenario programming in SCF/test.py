# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 23:25:27 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""

def product(*args, repeat):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = []
    for i in args:
        pools.append(tuple(i))
    pools = pools * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result
        

print(product([1,2,3], repeat=2))


