# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:10:50 2020

@author: zhen chen

MIT Licence.

Python version: 3.7


Description:  Python function for permutations of a given list 
    
"""


def product(args, repeat):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = [tuple(args)] * repeat
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return result
           