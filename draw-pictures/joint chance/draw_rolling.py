# -*- coding: utf-8 -*-
"""
Created on Mon May 16 20:07:18 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: draw the comparision results of rolling horizon framework
    
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


datas = pd.read_excel(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\RollingHorizonPerformance.xls')
df = datas[['demand mode', 'rolling length', 'sample one period', 'rolling time', 'rolling obj', 'rolling service rate']]
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df.drop(df[(df['rolling length'] == 2) & (df['sample one period'] == 5)].index, inplace = True)
df.drop(df[(df['rolling length'] == 3) & (df['sample one period'] == 5)].index, inplace = True)
df.drop(df[(df['rolling length'] == 3) & (df['sample one period'] == 9)].index, inplace = True)

df['rolling obj'] = df['rolling obj'] *100
df['rolling service rate'] = df['rolling service rate'] *100

df1 = df.groupby(['demand mode', 'rolling length', 'sample one period']).mean()
df1 = df1.reset_index()

plt.close('all')
fig1 = sns.lineplot(data=df1, x="demand mode", y="rolling obj", hue="rolling length")
plt.ylim([0, 100])

rolling_values1 = df1[df1['rolling length'] == 2].drop(columns = ['sample one period', 'rolling length', 'rolling time'])
rolling_values2 = df1[df1['rolling length'] == 3].drop(columns = ['sample one period', 'rolling length', 'rolling time'])
rolling_values3 = df1[df1['rolling length'] == 4].drop(columns = ['sample one period', 'rolling length', 'rolling time'])

header_txt = 'demand-mode rolling-obj2 rolling-service2 rolling-obj3 rolling-service3 rolling-obj4 rolling-service4'
output = rolling_values1.merge(rolling_values2, on = 'demand mode')
output = output.merge(rolling_values3, on = 'demand mode')
np.savetxt(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\rolling_output.txt', output.values, fmt='%.1f', header = header_txt, comments='', delimiter=' ')

