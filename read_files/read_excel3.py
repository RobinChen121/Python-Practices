# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:02:18 2023

@author: chen
"""


import numpy as np
import pandas as pd


datas = pd.read_excel('ServiceRateTest.xls')
df = datas[['demand mode', 'rolling obj', 'rolling service rate', 'Tom obj', 'sim Tom service', 'individual chance obj', 'sim individual chance service']]
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df1 = df.groupby(['demand mode']).mean()
df1 = df1 * 100;
df1.iloc[:, 4] = df1.iloc[:, 4] / 100;

print('\\addplot coordinates {', end='')
for i in range(10):
    print('(%d, %.3f) ' % (i+1, df1.iloc[i, 5]), end='')
print('};')