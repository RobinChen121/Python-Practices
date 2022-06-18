# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:48:38 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: the impact of holding cost
    
"""


import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


datas = pd.read_excel(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\JointChanceSAA5Periods-normalDist.xls')
df = datas[['demand mode', 'serviceRate', 'scenario number', 'iniCash', 'holdCost', 'price', 'sigmaCoe', 'sim SAA obj', 'sim SAA service rate', 'sim extend SAA obj', 'sim extend SAA service']]
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df['sim extend SAA service'] = df['sim extend SAA service'] *100
df['sim SAA service rate'] = df['sim SAA service rate'] *100
df['sim extend SAA obj'] = df['sim extend SAA obj'] *100
df['sim SAA obj'] = df['sim SAA obj'] *100

df1 = df[(df.serviceRate == 0.6)&(df['scenario number']==3125)]
df1 = df1[(df1.sigmaCoe == 0.25) & (df1.price == 16)]
df1 = df1[['demand mode', 'holdCost', 'sim extend SAA service', "sim SAA service rate", "sim extend SAA obj", "sim SAA obj"]].round(1)
df2 = df1.groupby(['holdCost', 'demand mode']).mean()
df2 = df2.reset_index()

plt.close('all')
fig1 = sns.lineplot(data=df2, x="demand mode", y="sim extend SAA service", hue="holdCost")
plt.ylim([0, 100])

fig2 = sns.lineplot(data=df2, x="demand mode", y="sim SAA service rate", hue="holdCost")
plt.ylim([0, 100])

holdCost_values1 = df2[df2['holdCost'] == 0].drop(columns = 'holdCost')
holdCost_values2 = df2[df2['holdCost'] == 0.5].drop(columns = 'holdCost')
holdCost_values3 = df2[df2['holdCost'] == 1].drop(columns = 'holdCost')

header_txt = 'demand-mode extend-hold0 SAA-hold0 extend-obj0 SAA-obj0 extend-hold0.5 SAA-hold0.5 extend-obj0.5 SAA-obj0.5 extend-hold1 SAA-hold1 extend-obj1 SAA-obj1'
output = holdCost_values1.merge(holdCost_values2, on = 'demand mode')
output = output.merge(holdCost_values3, on = 'demand mode')
np.savetxt(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\holdCost_output.txt', output.values, fmt='%.1f', header = header_txt, comments='', delimiter=' ')
