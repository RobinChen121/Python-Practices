#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:22:06 2022

@author: chen
"""

def part(n, k):
    def _part(n, k, max_value):
        if max_value <= -1:
            return []
        if k == 1:
            if n <= max_value:
                return [[n]]
            return []
        ret = []
        for i in range(0, n+1):
            ret += [[i] + sub for sub in _part(n-i, k-1, i)]
        return ret
    return _part(n, k, n)

print(part(3, 3))