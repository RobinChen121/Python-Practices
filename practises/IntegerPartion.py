# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:48:28 2019

@author: Zhen Chen

@email: okchen321@163.com 

Python version： 3.7

Description: partion an integer

"""
# 输出整数划分个数
def f1(n, m):
    if n == 0 or m == 0:
        return 0
    if n == 1 or m == 1 :
        return 1
    if m == n:        
        return 1 + f1(n, n-1)
    elif m > n:
        return f1(n, n)
    elif m < n:
        return f1(n-m, m) + f1(n, m-1)

# 输出具体的整数划分
def f2(n, m, string):
    if n == 0:
        print(string)
    else:
        if m > 1:
            f2(n, m - 1, string)
        if m <= n:
            f2(n - m, m, str(m)+ ' ' + string) 

n = 50; m = 3
print(f1(n, m))
f2(n, m, '')

