# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 22:49:35 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: draw bars for the joint chance results with confidence intervals for different scenario size
    
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


datas = pd.read_excel(r'/Users/chen/Documents/gitOld/Numerical-tests/joint chance datas/2022-08-03/eta_N.xls')
df = datas[['sampleNumSim', 'eta', 'SAAObj', 'scenarioObj', 'simSAAObj', 'simSAAService', 'simScenarioObj', 'simScenarioService']]
demandPatterns = ['size1', 'size2', 'size3', 'size4', 'size5']

# df1 = df[(df.eta==0.05) & (df.sampleNumSim==27)]

# row_num = df1.shape[0]
# for k in range(row_num):
#     print('%.2f\\\\ ' % df1.iloc[k, 3], end='') # saa
# print('\n')
# for k in range(row_num):
#     print('%.2f\\\\ ' % df1.iloc[k, 2], end='') # scenario

arr = []
for k in range(5):
    sample_size = (k*2+3) ** 3
    means = df[(df.eta==0.05) & (df.sampleNumSim==sample_size)].mean()
    arr.append([means[6], means[4], means[7], means[5]])

for k in range(len(arr[0])):
    print('\\addplot coordinates {', end='')
    for i in range(len(arr)):
        print('(%d, %.2f) ' % (i+1, arr[i][k]), end='')
    print('};')


        


