# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:48:03 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: the impact of required service level
    
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

df1 = df[(df.serviceRate == 0.6) | (df.serviceRate == 0.5) | (df.serviceRate == 0.7)]
df1 = df1[(df.sigmaCoe == 0.25) & (df.price == 16) & (df['scenario number']==3125) & (df['holdCost']==0)]
df1 = df1[['demand mode', 'serviceRate', 'sim extend SAA service', "sim SAA service rate", "sim extend SAA obj", "sim SAA obj"]].round(1)
df2 = df1.groupby(['demand mode', 'serviceRate']).mean()
df2 = df2.reset_index()

plt.close('all')
fig1 = sns.lineplot(data=df2, x="demand mode", y="sim extend SAA service", hue="serviceRate")
plt.ylim([0, 100])

fig2 = sns.lineplot(data=df2, x="demand mode", y="sim SAA service rate", hue="serviceRate")
plt.ylim([0, 100])

service_values1 = df2[df2['serviceRate'] == 0.5].drop(columns = 'serviceRate')
service_values2 = df2[df2['serviceRate'] == 0.6].drop(columns = 'serviceRate')
service_values3 = df2[df2['serviceRate'] == 0.7].drop(columns = 'serviceRate')

# np.savetxt(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\service_values0.5.txt', service_values1.values, fmt='%.1f', header = 'demand-mode SAA-service extent-service SAA-obj extend-obj', comments='', delimiter=' ')
# np.savetxt(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\service_values0.6.txt', service_values2.values, fmt='%.1f', header = 'demand-mode SAA-service extent-service SAA-obj extend-obj', comments='', delimiter=' ')
# np.savetxt(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\service_values0.7.txt', service_values3.values, fmt='%.1f', header = 'demand-mode SAA-service extent-service SAA-obj extend-obj', comments='', delimiter=' ')

header_txt = 'demand-mode extend-service0.5 SAA-service0.5 extend-obj0.5 SAA-obj0.5 extend-service0.6 SAA-service0.6 extend-obj0.6 SAA-obj0.6 extend-service0.7 SAA-service0.7 extend-obj0.7 SAA-obj0.7'
output = service_values1.merge(service_values2, on = 'demand mode')
output = output.merge(service_values3, on = 'demand mode')
np.savetxt(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\service_output.txt', output.values, fmt='%.1f', header = header_txt, comments='', delimiter=' ')

