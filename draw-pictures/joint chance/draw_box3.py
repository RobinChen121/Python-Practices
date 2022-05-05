# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:05:23 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: about the impact of different holding cost
    
"""
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
pd.options.mode.chained_assignment = None  # default='warn'


datas = pd.read_excel(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\JointChanceSAA5Periods-normalDist.xls')
df = datas[['demand mode', 'serviceRate', 'scenario number', 'iniCash', 'sigmaCoe', 'sim SAA obj', 'sim SAA service rate', 'sim extend SAA obj', 'sim extend SAA service', 'holdCost']]
demandPatterns = ['STA', 'LC1', 'LC2', 'SIN1', 'SIN2', 'RAND', 'EMP1', 'EMP2', 'EMP3', 'EMP4']

df['sim extend SAA service'] = df['sim extend SAA service'] *100
df['sim SAA service rate'] = df['sim SAA service rate'] *100
df['sim extend SAA obj'] = df['sim extend SAA obj'] *100
df['sim SAA obj'] = df['sim SAA obj'] *100

df1 = df[(df.serviceRate == 0.6) & (df.sigmaCoe == 0.25) & (df.iniCash == 40) & (df['scenario number']==3125)]
plt.close('all')
plt.figure()
fig1 = sns.boxplot(x="demand mode", y="sim extend SAA service", hue="holdCost",
                 data=df1, palette="Set3")
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('service rate')
plt.ylim([0, 100])
plt.axhline(y=60, xmin=0.01, xmax=0.99, ls='--', label = 'required service rate')
plt.legend()

plt.figure()
fig2 = sns.boxplot(x="demand mode", y="sim SAA service rate", hue="holdCost",
                 data=df1, palette="Set3")
plt.axhline(y=60, xmin=0.01, xmax=0.99, ls='--', label = 'required service rate')
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('service rate')
plt.ylim([0, 100])
fig2.legend()

plt.figure()
fig3 = sns.boxplot(x="demand mode", y="sim extend SAA obj", hue="holdCost", data=df1, palette="Set3")
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('survival probability')
plt.ylim([0, 100])
plt.legend()
# figure = fig3.get_figure() 
# figure.savefig(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\scenarioNum_extendObj.eps', dpi=3000)


plt.figure()
fig4 = sns.boxplot(x="demand mode", y="sim SAA obj", hue="holdCost",
                 data=df1, palette="Set3")
plt.xticks(np.arange(0, 10), demandPatterns)
plt.xlabel('demand patterns')
plt.ylabel('survival probability')
plt.ylim([0, 100])
plt.legend()
# figure = fig4.get_figure() 
# figure.savefig(r'D:\Users\chen_\git\Numerical-tests\joint chance datas\scenarioNum_SAAObj.eps', dpi=3000)