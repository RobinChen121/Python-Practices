# -*- coding: utf-8 -*-
"""
@date: Created on Fri Jul 27 20:07:02 2018

@author: Zhen Chen

@Python version: 3.6

@description: an example about class operator in python
    
"""


class Vector:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __str__(self):
        return "Vector (%d, %d)" % (self.a, self.b)

    # other operators are sub, mul, truediv, floordiv  mod
    def __add__(self, other):
        return Vector(self.a + other.a, self.b + other.b)

    def __mul__(self, other):
        return Vector(self.a * other.a, self.b * other.b)

    def __truediv__(self, other):
        return Vector(self.a / other.a, self.b / other.b)

    def __mod__(self, other):
        return Vector(self.a % other.a, self.b % other.b)


v1 = Vector(2, 10)
v2 = Vector(5, -2)
print(v1 + v2)
print(v1 * v2)
print(v1 / v2)
print(v1 % v2)
