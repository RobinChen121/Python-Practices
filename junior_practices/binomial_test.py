#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 18:11:35 2022

@author: chen
"""

import scipy.stats as st

for y in range(0, 130):
    expect_loss = 0
    for i in range(y - 96, y + 1):
        expect_loss += (i + 96 - y) * st.binom.pmf(i, y, 0.1)
    result = expect_loss * 15 + y - y * 0.1
    print(y, result + 151)
