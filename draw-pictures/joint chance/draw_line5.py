# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:52:17 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: the impact of margin
    
"""


import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


datas = pd.read_excel(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\JointChanceSAA5Periods-normalDist.xls')
df = datas[['demand mode', 'serviceRate', 'scenario number', 'iniCash', 'holdCost', 'sigmaCoe', 'price', 'sim SAA obj', 'sim SAA service rate', 'sim extend SAA obj', 'sim extend SAA service']]
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df['sim extend SAA service'] = df['sim extend SAA service'] *100
df['sim SAA service rate'] = df['sim SAA service rate'] *100
df['sim extend SAA obj'] = df['sim extend SAA obj'] *100
df['sim SAA obj'] = df['sim SAA obj'] *100

df1 = df[(df.serviceRate == 0.6)&(df['scenario number']==3125)]
df1 = df1[(df1.holdCost == 0) & (df1.sigmaCoe == 0.25)]
df1 = df1[['demand mode', 'price', 'sim extend SAA service', "sim SAA service rate", "sim extend SAA obj", "sim SAA obj"]].round(2)
df2 = df1.groupby(['demand mode', 'price']).mean()
df2 = df2.reset_index()

plt.close('all')
fig1 = sns.lineplot(data=df2, x="demand mode", y="sim extend SAA service", hue="price")
plt.ylim([0, 100])

fig2 = sns.lineplot(data=df2, x="demand mode", y="sim SAA service rate", hue="price")
plt.ylim([0, 100])

price_values1 = df2[df2['price'] == 16].drop(columns = 'price')
price_values2 = df2[df2['price'] == 20].drop(columns = 'price')
price_values3 = df2[df2['price'] == 26].drop(columns = 'price')

header_txt = 'demand-mode extend-price0.1 SAA-price0.1 extend-obj0.1 SAA-obj0.1 extend-price0.25 SAA-price0.25 extend-obj0.25 SAA-obj0.25 extend-price0.5 SAA-price0.5 extend-obj0.5 SAA-obj0.5'
output = price_values1.merge(price_values2, on = 'demand mode')
output = output.merge(price_values3, on = 'demand mode')
np.savetxt(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\price_output.txt', output.values, fmt='%.1f', header = header_txt, comments='', delimiter=' ')

