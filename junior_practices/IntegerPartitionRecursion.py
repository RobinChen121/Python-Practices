# -*- coding: utf-8 -*-
"""
@date: Created on Fri Jul 27 20:01:45 2018

@author: Zhen Chen

@Python version: 3.6

@description: a recursion about integer partitioning
    
"""


# output only the number of partitioning
def f1(n, m):
    if n == 0 or m == 0:
        return 0
    if n == 1 or m == 1:
        return 1
    if m == n:
        return 1 + f1(n, n - 1)
    elif m > n:
        return f1(n, n)
    elif m < n:
        return f1(n - m, m) + f1(n, m - 1)


# output detailed partitioning
def f2(n, m, string):
    if n == 0:
        print(string)
    else:
        if m > 1:
            f2(n, m - 1, string)
        if m <= n:
            f2(n - m, m, str(m) + " " + string)


n = 5
m = 4
print(f1(n, m))
f2(n, m, "")
