# -*- coding: utf-8 -*-
"""
Created on Wed May 18 13:01:10 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: draw the rolling results for large planning horizon problem of 12 periods
    
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


datas = pd.read_excel(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\RollingHorizonPerformance2.xlsx')
df = datas[['demand mode', 'rolling time', 'rolling obj', 'rolling service rate']]
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df['rolling obj'] = df['rolling obj'] *100
df['rolling service rate'] = df['rolling service rate'] *100

df1 = df.groupby('demand mode').mean()
df1 = df1.reset_index()

sns.set()
fig1 = sns.lineplot(data=df1, x="demand mode", y="rolling obj")
sns.lineplot(data=df1, x="demand mode", y="rolling service rate")

header_txt = 'demand-mode rolling-time rolling-obj rolling-service-rate'
np.savetxt(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\rolling_output2.txt', df1.values, fmt='%.1f', header = header_txt, comments='', delimiter=' ')
